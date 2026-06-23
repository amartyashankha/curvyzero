"""Profile-only mock collector for batched CurvyTron observations.

This module does not call LightZero and does not change trainer defaults. It is
the next bridge after the batched renderer boundary: take a vector CurvyTron
step, produce row/player policy stacks, materialize LightZero-shaped scalar
rows, and optionally run the current RND reward-model hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from curvyzero.training import exploration_bonus as xb
from curvyzero.training.source_state_batched_observation_profile import (
    SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS,
)
from curvyzero.training.source_state_batched_observation_profile import (
    SourceStateBatchedObservationProfileFacade,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)


SCHEMA_ID = "curvyzero_source_state_batched_observation_mock_collector_profile/v0"
_MISSING = object()
_ACTION_COUNT = 3
_STACK_SHAPE = (4, 64, 64)


class _FallbackDiscreteActionSpace:
    def __init__(self, n: int) -> None:
        self.n = int(n)
        self._rng = np.random.default_rng(0)

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self) -> int:
        return int(self._rng.integers(0, self.n))


def _make_discrete_action_space(n: int) -> Any:
    try:
        import gym
    except Exception:
        gym = None
    if gym is not None:
        return gym.spaces.Discrete(int(n))
    return _FallbackDiscreteActionSpace(int(n))


def _make_box_space(*, low: float, high: float, shape: tuple[int, ...], dtype: Any) -> Any:
    try:
        import gym
    except Exception:
        gym = None
    if gym is not None:
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return SimpleNamespace(low=low, high=high, shape=shape, dtype=dtype)


@dataclass(frozen=True, slots=True)
class MockCollectorConfig:
    batch_size: int = 8
    player_count: int = 2
    steps: int = 8
    warmup_steps: int = 2
    seed: int = 0
    max_ticks: int = 2_000
    include_rnd_meter: bool = False
    rnd_batch_size: int = 64
    rnd_update_per_collect: int = xb.RND_DEFAULT_UPDATE_PER_COLLECT
    rnd_device: str = "cpu"
    pickle_payload: bool = True


@dataclass(frozen=True, slots=True)
class MockBaseEnvTimestep:
    obs: dict[str, Any]
    reward: np.ndarray
    done: np.ndarray
    info: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class BatchedSurfaceProfileLoopStep:
    surface_step: Any
    timestep: MockBaseEnvTimestep
    flat_obs: np.ndarray
    target_reward: np.ndarray


@dataclass(frozen=True, slots=True)
class ScalarActionBridgeOutput:
    """LightZero-env-manager-shaped output for the profile-only bridge."""

    ready_obs: dict[int, dict[str, Any]]
    timestep_by_env_id: dict[int, MockBaseEnvTimestep]
    policy_env_id: np.ndarray
    policy_env_row: np.ndarray
    policy_player: np.ndarray
    surface_step: Any
    autoreset_row_mask: np.ndarray
    profile_timing_sec: dict[str, float] | None = None
    profile_counts: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class ProfileEnvManagerStepResult:
    """Manager-shaped profile result after one scalar action dict step."""

    timestep_by_env_id: dict[int, MockBaseEnvTimestep]
    ready_obs: dict[int, dict[str, Any]]
    bridge_output: ScalarActionBridgeOutput


class BatchedSourceStateTrainerProfileLoop:
    """Profile-only loop that keeps the trainer surface batched until scalar output."""

    profile_only = True
    stock_lightzero_integrated = False
    touches_live_runs = False

    def __init__(self, surface: SourceStateMultiplayerTrainerSurface) -> None:
        self.surface = surface
        self.batch_size = int(surface.batch_size)
        self.player_count = int(surface.player_count)

    def reset(self, seed: int | np.ndarray | None = None) -> BatchedSurfaceProfileLoopStep:
        return self._materialize(self.surface.reset(seed=seed))

    def step(
        self,
        joint_action: np.ndarray,
        *,
        timer_advance_ms: float | np.ndarray | None = None,
    ) -> BatchedSurfaceProfileLoopStep:
        return self._materialize(
            self.surface.step(joint_action, timer_advance_ms=timer_advance_ms)
        )

    def _materialize(self, surface_step: Any) -> BatchedSurfaceProfileLoopStep:
        timestep, flat_obs, target_reward = materialize_trainer_surface_policy_timestep(
            surface_step=surface_step,
            batch_size=self.batch_size,
            player_count=self.player_count,
        )
        return BatchedSurfaceProfileLoopStep(
            surface_step=surface_step,
            timestep=timestep,
            flat_obs=flat_obs,
            target_reward=target_reward,
        )


class BatchedLightZeroScalarActionBridge:
    """Profile-only bridge from scalar LightZero actions to one batched CurvyTron step.

    LightZero's collectors think in scalar env ids. CurvyTron wants one joint
    action per physical row. This bridge is the smallest local canary for that
    boundary: accept one action per live ``scalar_env_id``, commit one vector
    step, then expose scalar timesteps again.
    """

    profile_only = True
    stock_lightzero_integrated = False
    touches_live_runs = False

    def __init__(
        self,
        surface: SourceStateMultiplayerTrainerSurface,
        *,
        autoreset_terminal_rows: bool = True,
        timer_advance_ms: float | np.ndarray | None = None,
    ) -> None:
        self.loop = BatchedSourceStateTrainerProfileLoop(surface)
        self.batch_size = int(surface.batch_size)
        self.player_count = int(surface.player_count)
        self.autoreset_terminal_rows = bool(autoreset_terminal_rows)
        self.timer_advance_ms = timer_advance_ms
        self._last_output: ScalarActionBridgeOutput | None = None

    @property
    def ready_env_ids(self) -> tuple[int, ...]:
        if self._last_output is None:
            return ()
        return tuple(int(item) for item in self._last_output.policy_env_id)

    def scalar_env_id(self, *, row: int, player: int) -> int:
        row_int = int(row)
        player_int = int(player)
        if row_int < 0 or row_int >= self.batch_size:
            raise ValueError("row is out of range")
        if player_int < 0 or player_int >= self.player_count:
            raise ValueError("player is out of range")
        return row_int * self.player_count + player_int

    def row_player_for_scalar_env_id(self, env_id: int) -> tuple[int, int]:
        env_int = int(env_id)
        max_env_id = self.batch_size * self.player_count
        if env_int < 0 or env_int >= max_env_id:
            raise ValueError("scalar env id is out of range")
        return divmod(env_int, self.player_count)

    def reset(self, seed: int | np.ndarray | None = None) -> ScalarActionBridgeOutput:
        output = self._output_from_loop_step(
            self.loop.reset(seed=seed),
            autoreset_row_mask=np.zeros(self.batch_size, dtype=bool),
        )
        self._last_output = output
        return output

    def step(self, action_by_env_id: Mapping[int, Any]) -> ScalarActionBridgeOutput:
        if self._last_output is None:
            raise RuntimeError("reset must be called before step")
        timing: dict[str, float] = {}
        counts: dict[str, int] = {}
        started = time.perf_counter()
        action_env_ids = np.asarray(sorted(int(key) for key in action_by_env_id), dtype=np.int32)
        timing["action_env_id_sort_sec"] = time.perf_counter() - started
        counts["action_count"] = int(action_env_ids.size)

        started = time.perf_counter()
        actions = self._joint_action_from_scalar_actions(action_by_env_id)
        timing["joint_action_from_scalar_sec"] = time.perf_counter() - started

        started = time.perf_counter()
        loop_step = self.loop.step(actions, timer_advance_ms=self.timer_advance_ms)
        timing["loop_step_sec"] = time.perf_counter() - started
        autoreset_mask = np.zeros(self.batch_size, dtype=bool)
        output_step = loop_step
        if self.autoreset_terminal_rows:
            autoreset_mask = np.asarray(loop_step.surface_step.done, dtype=bool).copy()
            if autoreset_mask.shape != (self.batch_size,):
                raise ValueError("surface done mask has wrong shape")
            if bool(autoreset_mask.any()):
                started = time.perf_counter()
                reset_step = self.loop.surface.reset(row_mask=autoreset_mask)
                output_step = self.loop._materialize(reset_step)
                timing["autoreset_reset_materialize_sec"] = time.perf_counter() - started
        counts["autoreset_row_count"] = int(np.asarray(autoreset_mask, dtype=bool).sum())
        started = time.perf_counter()
        output = self._output_from_loop_step(
            output_step,
            timestep_source=loop_step,
            timestep_env_ids=action_env_ids,
            autoreset_row_mask=autoreset_mask,
            profile_timing_sec=timing,
            profile_counts=counts,
        )
        if output.profile_timing_sec is not None:
            output.profile_timing_sec["output_from_loop_step_sec"] = (
                time.perf_counter() - started
            )
        self._last_output = output
        return output

    def _joint_action_from_scalar_actions(self, action_by_env_id: Mapping[int, Any]) -> np.ndarray:
        ready_ids = set(self.ready_env_ids)
        provided_ids = {int(key) for key in action_by_env_id}
        missing = sorted(ready_ids - provided_ids)
        extra = sorted(provided_ids - ready_ids)
        partial_missing_rows: list[int] = []
        if missing:
            missing_by_row: dict[int, set[int]] = {}
            for env_id in missing:
                row, player = self.row_player_for_scalar_env_id(env_id)
                missing_by_row.setdefault(row, set()).add(player)
            partial_missing_rows = sorted(
                row
                for row, players in missing_by_row.items()
                if len(players) != self.player_count
            )
        if extra or partial_missing_rows:
            raise ValueError(
                "action_by_env_id must contain complete physical CurvyTron rows; "
                f"missing={missing}, partial_missing_rows={partial_missing_rows}, "
                f"extra={extra}"
            )
        joint_action = np.ones((self.batch_size, self.player_count), dtype=np.int16)
        for env_id, raw_action in action_by_env_id.items():
            row, player = self.row_player_for_scalar_env_id(int(env_id))
            action = _coerce_scalar_action(raw_action)
            if action < 0 or action >= 3:
                raise ValueError("actions must be in [0, 2]")
            joint_action[row, player] = action
        return joint_action

    def _output_from_loop_step(
        self,
        loop_step: BatchedSurfaceProfileLoopStep,
        *,
        timestep_source: BatchedSurfaceProfileLoopStep | None = None,
        timestep_env_ids: np.ndarray | None = None,
        autoreset_row_mask: np.ndarray,
        profile_timing_sec: dict[str, float] | None = None,
        profile_counts: dict[str, int] | None = None,
    ) -> ScalarActionBridgeOutput:
        timing = {} if profile_timing_sec is None else dict(profile_timing_sec)
        counts = {} if profile_counts is None else dict(profile_counts)
        started = time.perf_counter()
        policy_env_row = np.asarray(loop_step.surface_step.policy_env_row, dtype=np.int32)
        policy_player = np.asarray(loop_step.surface_step.policy_player, dtype=np.int16)
        policy_env_id = (
            policy_env_row.astype(np.int64) * self.player_count
            + policy_player.astype(np.int64)
        ).astype(np.int32, copy=False)
        timing["policy_env_id_build_sec"] = time.perf_counter() - started

        started = time.perf_counter()
        ready_obs = _ready_obs_by_env_id(
            env_ids=policy_env_id,
            timestep=loop_step.timestep,
        )
        timing["ready_obs_by_env_id_sec"] = time.perf_counter() - started
        counts["ready_obs_count"] = int(len(ready_obs))

        source = loop_step if timestep_source is None else timestep_source
        if timestep_env_ids is None:
            timestep_env_id = policy_env_id
            timestep = source.timestep
        else:
            started = time.perf_counter()
            timestep_env_id = np.asarray(timestep_env_ids, dtype=np.int32)
            source_env_row = np.asarray(source.surface_step.policy_env_row, dtype=np.int32)
            source_player = np.asarray(source.surface_step.policy_player, dtype=np.int16)
            source_env_id = (
                source_env_row.astype(np.int64) * self.player_count
                + source_player.astype(np.int64)
            ).astype(np.int32, copy=False)
            timing["timestep_env_id_prepare_sec"] = time.perf_counter() - started
            if (
                timestep_env_id.shape == source_env_id.shape
                and bool(np.array_equal(timestep_env_id, source_env_id))
            ):
                timestep = source.timestep
            else:
                started = time.perf_counter()
                timestep, _flat_obs, _target_reward = materialize_trainer_surface_env_id_timestep(
                    surface_step=source.surface_step,
                    env_ids=timestep_env_id,
                    batch_size=self.batch_size,
                    player_count=self.player_count,
                )
                timing["env_id_timestep_materialize_sec"] = time.perf_counter() - started
        started = time.perf_counter()
        timestep_by_env_id = _split_timestep_by_env_id(
            env_ids=timestep_env_id,
            timestep=timestep,
        )
        timing["split_timestep_by_env_id_sec"] = time.perf_counter() - started
        counts.update(_timestep_materialization_counts(timestep))
        counts["timestep_count"] = int(len(timestep_by_env_id))
        return ScalarActionBridgeOutput(
            ready_obs=ready_obs,
            timestep_by_env_id=timestep_by_env_id,
            policy_env_id=policy_env_id.copy(),
            policy_env_row=policy_env_row.copy(),
            policy_player=policy_player.copy(),
            surface_step=loop_step.surface_step,
            autoreset_row_mask=np.asarray(autoreset_row_mask, dtype=bool).copy(),
            profile_timing_sec=timing,
            profile_counts=counts,
        )


class BatchedLightZeroProfileEnvManager:
    """Tiny profile-only env-manager facade around the batched scalar bridge."""

    profile_only = True
    stock_lightzero_integrated = False
    touches_live_runs = False

    def __init__(self, bridge: BatchedLightZeroScalarActionBridge) -> None:
        self.bridge = bridge
        self.env_num = bridge.batch_size * bridge.player_count
        self._closed = False
        self._seed: int | None = None
        self._last_output: ScalarActionBridgeOutput | None = None
        self.last_reset_info: dict[int, dict[str, Any]] = {}

    @property
    def ready_obs(self) -> dict[int, dict[str, Any]]:
        if self._last_output is None:
            return {}
        return self._last_output.ready_obs

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        del dynamic_seed
        self._seed = int(seed)

    def reset(self, reset_param: Mapping[int, Any] | int | None = None) -> None:
        self._raise_if_closed()
        seed = self._seed
        if isinstance(reset_param, Mapping):
            if reset_param:
                env_ids = sorted(int(key) for key in reset_param)
                if env_ids != list(range(self.env_num)):
                    raise ValueError(
                        "profile env manager reset currently requires all scalar env ids"
                    )
                first = reset_param.get(0)
                if isinstance(first, Mapping) and "seed" in first:
                    seed = int(first["seed"])
        elif reset_param is not None:
            seed = int(reset_param)
        output = self.bridge.reset(seed=seed)
        self._last_output = output
        self.last_reset_info = {
            env_id: {
                "profile_only": True,
                "stock_lightzero_integrated": False,
                "vector_surface_batch_size": self.bridge.batch_size,
                "scalar_env_id": int(env_id),
                "row": int(env_id) // self.bridge.player_count,
                "player": int(env_id) % self.bridge.player_count,
            }
            for env_id in output.ready_obs
        }

    def step(self, actions: Mapping[int, Any]) -> ProfileEnvManagerStepResult:
        self._raise_if_closed()
        output = self.bridge.step(actions)
        self._last_output = output
        return ProfileEnvManagerStepResult(
            timestep_by_env_id=output.timestep_by_env_id,
            ready_obs=output.ready_obs,
            bridge_output=output,
        )

    def close(self) -> None:
        self._closed = True

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("profile env manager is closed")


class BatchedLightZeroStockEnvManagerAdapter:
    """Stock-shaped profile manager that returns ``env_id -> timestep`` mappings.

    The local profile manager keeps richer debug output around each step. Stock
    DI-engine collectors usually expect ``step`` to return only a mapping from
    scalar env id to a ``BaseEnvTimestep``-like object. This adapter is the
    smallest reversible boundary for the stock ``train_muzero`` canary.
    """

    profile_only = True
    stock_lightzero_integrated = "profile_canary_only"
    touches_live_runs = False

    def __init__(
        self,
        bridge: BatchedLightZeroScalarActionBridge,
        *,
        base_env_timestep_cls: Any | None = None,
    ) -> None:
        self._profile_manager = BatchedLightZeroProfileEnvManager(bridge)
        self._base_env_timestep_cls = base_env_timestep_cls
        self.env_num = self._profile_manager.env_num
        self.last_profile_step: ProfileEnvManagerStepResult | None = None
        self._action_space = _make_discrete_action_space(_ACTION_COUNT)
        self._observation_space = _make_box_space(
            low=0.0,
            high=1.0,
            shape=_STACK_SHAPE,
            dtype=np.float32,
        )
        self._reward_space = _make_box_space(
            low=-float("inf"),
            high=float("inf"),
            shape=(),
            dtype=np.float32,
        )
        self._env_ref = SimpleNamespace(
            action_space=self._action_space,
            observation_space=self._observation_space,
            reward_space=self._reward_space,
            enable_save_replay=lambda *_args, **_kwargs: None,
        )
        self._episode_return_by_env_id = {env_id: 0.0 for env_id in range(self.env_num)}

    @property
    def ready_obs(self) -> dict[int, dict[str, Any]]:
        return self._profile_manager.ready_obs

    @property
    def ready_obs_id(self) -> list[int]:
        return sorted(int(env_id) for env_id in self.ready_obs)

    @property
    def last_reset_info(self) -> dict[int, dict[str, Any]]:
        return self._profile_manager.last_reset_info

    @property
    def env_ref(self) -> Any:
        return self._env_ref

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def reward_space(self) -> Any:
        return self._reward_space

    @property
    def done(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return bool(self._profile_manager._closed)

    @property
    def active_env(self) -> list[int]:
        return list(range(self.env_num))

    def launch(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        if not self.ready_obs:
            self.reset()

    def seed(self, seed: int | list[int] | Mapping[int, int], dynamic_seed: bool = True) -> None:
        if isinstance(seed, Mapping):
            seed = next(iter(seed.values())) if seed else 0
        elif isinstance(seed, list):
            seed = seed[0] if seed else 0
        self._profile_manager.seed(seed, dynamic_seed=dynamic_seed)
        try:
            self._action_space.seed(int(seed))
        except Exception:
            pass

    def reset(self, reset_param: Mapping[int, Any] | int | None = None) -> None:
        self._profile_manager.reset(reset_param)
        for env_id in self.ready_obs:
            self._episode_return_by_env_id[int(env_id)] = 0.0

    def step(self, actions: Mapping[int, Any]) -> dict[int, Any]:
        result = self._profile_manager.step(actions)
        self.last_profile_step = result
        timesteps = {}
        conversion_started = time.perf_counter()
        for env_id, timestep in result.timestep_by_env_id.items():
            env_int = int(env_id)
            reward = float(np.asarray(timestep.reward, dtype=np.float32).reshape(()).item())
            done = bool(np.asarray(timestep.done, dtype=np.bool_).reshape(()).item())
            episode_return = float(self._episode_return_by_env_id.get(env_int, 0.0) + reward)
            self._episode_return_by_env_id[env_int] = 0.0 if done else episode_return
            timesteps[env_int] = _to_stock_base_env_timestep(
                timestep=timestep,
                base_env_timestep_cls=self._base_env_timestep_cls,
                reward=reward,
                done=done,
                info_overrides=(
                    {"eval_episode_return": episode_return}
                    if done
                    else None
                ),
            )
        bridge_output = result.bridge_output
        if bridge_output.profile_timing_sec is not None:
            bridge_output.profile_timing_sec["stock_base_env_timestep_conversion_sec"] = (
                time.perf_counter() - conversion_started
            )
        return timesteps

    def close(self) -> None:
        self._profile_manager.close()

    def enable_save_replay(self, replay_path: list[str] | str) -> None:
        del replay_path

    def random_action(self) -> dict[int, int]:
        return {env_id: int(self._action_space.sample()) for env_id in self.ready_obs_id}


def run_mock_collector_profile(config: MockCollectorConfig) -> dict[str, Any]:
    """Run a local, profile-only collector-shaped loop."""

    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.player_count < 2:
        raise ValueError("player_count must be at least 2")
    if config.steps < 1:
        raise ValueError("steps must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=config.batch_size,
        player_count=config.player_count,
        player_view_mode=SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS,
        seed=config.seed,
        max_ticks=config.max_ticks,
    )
    rng = np.random.default_rng(config.seed + 101)
    rnd_model = _build_rnd_model(config) if config.include_rnd_meter else None

    timings = {
        "facade_step_sec": 0.0,
        "scalarize_sec": 0.0,
        "pickle_sec": 0.0,
        "rnd_collect_data_sec": 0.0,
        "rnd_train_with_data_sec": 0.0,
        "rnd_estimate_sec": 0.0,
    }
    bytes_total = 0
    done_rows = 0
    rows_per_step = config.batch_size * config.player_count
    materialization_counts = {
        "flat_obs_nbytes": 0,
        "action_mask_nbytes": 0,
        "reward_nbytes": 0,
        "done_nbytes": 0,
        "final_observation_nbytes": 0,
        "info_count": 0,
        "materialized_timestep_count": 0,
    }

    facade.reset(seed=config.seed)
    started_total = time.perf_counter()
    for iteration in range(config.warmup_steps + config.steps):
        actions = rng.integers(0, 3, size=config.batch_size, dtype=np.int16)
        other_actions = rng.integers(0, 3, size=config.batch_size, dtype=np.int16)

        started = time.perf_counter()
        step = facade.step(actions=actions, other_actions=other_actions)
        facade_step_sec = _elapsed(started)

        started = time.perf_counter()
        timestep, flat_obs, target_reward = materialize_lightzero_scalar_timestep(
            step_observation=step.observation,
            step_reward=step.reward,
            step_done=step.done,
            final_observation=step.final_observation,
            batch_size=config.batch_size,
            player_count=config.player_count,
        )
        scalarize_sec = _elapsed(started)
        step_materialization_counts = _timestep_materialization_counts(timestep)

        pickle_sec = 0.0
        pickle_bytes = 0
        if config.pickle_payload:
            started = time.perf_counter()
            payload = pickle.dumps(timestep, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_sec = _elapsed(started)
            pickle_bytes = len(payload)

        if iteration < config.warmup_steps:
            continue

        timings["facade_step_sec"] += facade_step_sec
        timings["scalarize_sec"] += scalarize_sec
        timings["pickle_sec"] += pickle_sec
        bytes_total += int(pickle_bytes)
        done_rows += int(timestep.done.sum())
        for key, value in step_materialization_counts.items():
            materialization_counts[key] += int(value)

        if rnd_model is not None:
            segment = SimpleNamespace(obs_segment=flat_obs)
            started = time.perf_counter()
            rnd_model.collect_data([[segment]])
            timings["rnd_collect_data_sec"] += _elapsed(started)

            started = time.perf_counter()
            rnd_model.train_with_data()
            timings["rnd_train_with_data_sec"] += _elapsed(started)

            started = time.perf_counter()
            rnd_model.estimate([[flat_obs], [target_reward]])
            timings["rnd_estimate_sec"] += _elapsed(started)

    total_sec = _elapsed(started_total)
    measured_rows = rows_per_step * config.steps
    rnd_metrics = None if rnd_model is None else rnd_model.metrics_snapshot(reason="mock_profile")
    return {
        "schema_id": SCHEMA_ID,
        "profile_only": True,
        "stock_lightzero_integrated": False,
        "trainer_defaults_changed": False,
        "semantic_contract": {
            "pixel_exact_required": False,
            "required": [
                "browser_lines_plus_simple_symbols_information",
                "row_player_order",
                "player_perspective",
                "stack_fifo_newest_last",
                "terminal_final_observation_before_reset",
                "no_hidden_cpu_fallback_for_gpu_candidate",
                "rnd_latest_frame_matches_policy_stack",
            ],
        },
        "config": {
            "batch_size": config.batch_size,
            "player_count": config.player_count,
            "steps": config.steps,
            "warmup_steps": config.warmup_steps,
            "seed": config.seed,
            "max_ticks": config.max_ticks,
            "include_rnd_meter": config.include_rnd_meter,
            "rnd_batch_size": config.rnd_batch_size,
            "rnd_update_per_collect": config.rnd_update_per_collect,
            "rnd_device": config.rnd_device,
            "pickle_payload": config.pickle_payload,
        },
        "rows_per_step": rows_per_step,
        "measured_rows": measured_rows,
        "total_sec": total_sec,
        "rows_per_sec": (float(measured_rows) / total_sec) if total_sec > 0.0 else None,
        "timings": timings,
        "timing_per_row_sec": {
            key: (value / float(measured_rows)) if measured_rows else None
            for key, value in timings.items()
        },
        "pickle_bytes_total": bytes_total,
        "pickle_bytes_per_row": (float(bytes_total) / float(measured_rows)) if measured_rows else 0.0,
        **materialization_counts,
        "done_rows": done_rows,
        "rnd_metrics": rnd_metrics,
    }


def materialize_lightzero_scalar_timestep(
    *,
    step_observation: np.ndarray,
    step_reward: np.ndarray,
    step_done: np.ndarray,
    final_observation: np.ndarray | None,
    action_mask: np.ndarray | None = None,
    batch_size: int,
    player_count: int,
) -> tuple[MockBaseEnvTimestep, np.ndarray, np.ndarray]:
    raw_obs = np.asarray(step_observation)
    obs = raw_obs.astype(np.float32, copy=False)
    if raw_obs.dtype == np.uint8:
        obs = obs * np.float32(1.0 / 255.0)
    expected_obs_shape = (batch_size, player_count, 4, 64, 64)
    if obs.shape != expected_obs_shape:
        raise ValueError(f"step_observation must have shape {expected_obs_shape}, got {obs.shape}")
    flat_obs = np.ascontiguousarray(obs.reshape(batch_size * player_count, 4, 64, 64))

    reward = _scalarize_reward(step_reward, batch_size=batch_size, player_count=player_count)
    done = np.repeat(np.asarray(step_done, dtype=np.bool_), player_count)
    if action_mask is None:
        flat_action_mask = np.ones((batch_size * player_count, _ACTION_COUNT), dtype=np.bool_)
    else:
        mask = np.asarray(action_mask, dtype=np.bool_)
        expected_batch_mask_shape = (batch_size, player_count, _ACTION_COUNT)
        expected_flat_mask_shape = (batch_size * player_count, _ACTION_COUNT)
        if mask.shape == expected_batch_mask_shape:
            flat_action_mask = np.ascontiguousarray(mask.reshape(expected_flat_mask_shape))
        elif mask.shape == expected_flat_mask_shape:
            flat_action_mask = np.ascontiguousarray(mask)
        else:
            raise ValueError(
                "action_mask must have shape "
                f"{expected_batch_mask_shape} or {expected_flat_mask_shape}, got {mask.shape}"
            )
    rows = np.repeat(np.arange(batch_size, dtype=np.int32), player_count)
    players = np.tile(np.arange(player_count, dtype=np.int32), batch_size)
    final_array = None
    if final_observation is not None:
        raw_final_array = np.asarray(final_observation)
        final_array = raw_final_array.astype(np.float32, copy=False)
        if raw_final_array.dtype == np.uint8:
            final_array = final_array * np.float32(1.0 / 255.0)
        if final_array.shape != expected_obs_shape:
            raise ValueError(
                f"final_observation must have shape {expected_obs_shape}, "
                f"got {final_array.shape}"
            )
    info = []
    for index, (row, player) in enumerate(zip(rows, players, strict=True)):
        item: dict[str, Any] = {
            "row": int(row),
            "player": int(player),
            "player_view": int(player),
            "profile_only": True,
            "final_observation_present": final_array is not None and bool(done[index]),
        }
        if final_array is not None and bool(done[index]):
            item["final_observation"] = final_array[int(row), int(player)].copy()
        info.append(item)
    timestep = MockBaseEnvTimestep(
        obs={
            "observation": flat_obs,
            "action_mask": flat_action_mask,
            "to_play": np.full((batch_size * player_count,), -1, dtype=np.int64),
        },
        reward=reward,
        done=done,
        info=info,
    )
    target_reward = reward.reshape(batch_size * player_count, 1).astype(np.float32, copy=False)
    return timestep, flat_obs, target_reward


def materialize_trainer_surface_policy_timestep(
    *,
    surface_step: Any,
    batch_size: int,
    player_count: int,
) -> tuple[MockBaseEnvTimestep, np.ndarray, np.ndarray]:
    flat_obs = np.ascontiguousarray(np.asarray(surface_step.policy_observation, dtype=np.float32))
    if flat_obs.ndim != 4 or flat_obs.shape[1:] != (4, 64, 64):
        raise ValueError(
            "surface_step.policy_observation must have shape [N,4,64,64], "
            f"got {flat_obs.shape}"
        )
    rows = np.asarray(surface_step.policy_env_row, dtype=np.int32)
    players = np.asarray(surface_step.policy_player, dtype=np.int16)
    if rows.shape != (flat_obs.shape[0],) or players.shape != (flat_obs.shape[0],):
        raise ValueError("policy row/player arrays must match policy observation rows")
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= batch_size):
        raise ValueError("policy_env_row contains out-of-range rows")
    if players.size and (int(players.min()) < 0 or int(players.max()) >= player_count):
        raise ValueError("policy_player contains out-of-range players")

    action_mask = np.asarray(surface_step.policy_action_mask, dtype=np.bool_)
    if action_mask.shape != (flat_obs.shape[0], 3):
        raise ValueError(f"policy_action_mask must have shape [N,3], got {action_mask.shape}")

    reward_array = np.asarray(surface_step.reward, dtype=np.float32)
    if reward_array.shape != (batch_size, player_count):
        raise ValueError(
            f"surface_step.reward must have shape ({batch_size}, {player_count}), "
            f"got {reward_array.shape}"
        )
    reward = np.ascontiguousarray(reward_array[rows, players].astype(np.float32, copy=False))
    done_by_row = np.asarray(surface_step.done, dtype=np.bool_)
    if done_by_row.shape != (batch_size,):
        raise ValueError(f"surface_step.done must have shape ({batch_size},), got {done_by_row.shape}")
    done = np.ascontiguousarray(done_by_row[rows])
    final_row_mask = np.asarray(
        getattr(surface_step, "final_observation_row_mask", np.zeros(batch_size, dtype=bool)),
        dtype=np.bool_,
    )
    if final_row_mask.shape != (batch_size,):
        raise ValueError(
            f"surface_step.final_observation_row_mask must have shape ({batch_size},), "
            f"got {final_row_mask.shape}"
        )
    final_present = bool(final_row_mask.any())

    expected_final_shape = (batch_size, player_count, 4, 64, 64)
    raw_final_observation = getattr(surface_step, "final_observation", _MISSING)
    if raw_final_observation is _MISSING:
        if final_present:
            raise ValueError(
                "surface_step.final_observation is required when "
                "final_observation_row_mask contains terminal rows"
            )
        final_observation = np.zeros(expected_final_shape, dtype=np.float32)
    else:
        final_observation = np.asarray(raw_final_observation, dtype=np.float32)
        if final_observation.shape != expected_final_shape:
            raise ValueError(
                f"surface_step.final_observation must have shape {expected_final_shape}, "
                f"got {final_observation.shape}"
            )

    expected_final_reward_shape = (batch_size, player_count)
    raw_final_reward_map = getattr(surface_step, "final_reward_map", _MISSING)
    if raw_final_reward_map is _MISSING:
        if final_present:
            raise ValueError(
                "surface_step.final_reward_map is required when "
                "final_observation_row_mask contains terminal rows"
            )
        final_reward_map = np.zeros(expected_final_reward_shape, dtype=np.float32)
    else:
        final_reward_map = np.asarray(raw_final_reward_map, dtype=np.float32)
        if final_reward_map.shape != expected_final_reward_shape:
            raise ValueError(
                "surface_step.final_reward_map must have shape "
                f"{expected_final_reward_shape}, got {final_reward_map.shape}"
            )

    info = []
    for row, player in zip(rows, players, strict=True):
        final_present = bool(final_row_mask[int(row)])
        item: dict[str, Any] = {
            "row": int(row),
            "player": int(player),
            "player_view": int(player),
            "profile_only": True,
            "surface_profile_loop": True,
            "final_observation_present": final_present,
        }
        if final_present:
            item["final_observation"] = final_observation[int(row), int(player)].copy()
            item["final_reward"] = float(final_reward_map[int(row), int(player)])
        info.append(item)
    timestep = MockBaseEnvTimestep(
        obs={
            "observation": flat_obs,
            "action_mask": action_mask,
            "to_play": np.full((flat_obs.shape[0],), -1, dtype=np.int64),
        },
        reward=reward,
        done=done,
        info=info,
    )
    target_reward = reward.reshape(flat_obs.shape[0], 1).astype(np.float32, copy=False)
    return timestep, flat_obs, target_reward


def materialize_trainer_surface_env_id_timestep(
    *,
    surface_step: Any,
    env_ids: np.ndarray,
    batch_size: int,
    player_count: int,
) -> tuple[MockBaseEnvTimestep, np.ndarray, np.ndarray]:
    env_id_array = np.asarray(env_ids, dtype=np.int32).reshape(-1)
    if env_id_array.size:
        max_env_id = batch_size * player_count
        if int(env_id_array.min()) < 0 or int(env_id_array.max()) >= max_env_id:
            raise ValueError("env_ids contain out-of-range scalar env ids")
    rows = (env_id_array // player_count).astype(np.int32, copy=False)
    players = (env_id_array % player_count).astype(np.int16, copy=False)

    observation_array = np.asarray(surface_step.observation, dtype=np.float32)
    expected_obs_shape = (batch_size, player_count, 4, 64, 64)
    if observation_array.shape != expected_obs_shape:
        raise ValueError(
            f"surface_step.observation must have shape {expected_obs_shape}, "
            f"got {observation_array.shape}"
        )
    flat_obs = np.ascontiguousarray(observation_array[rows, players])

    action_mask_array = np.asarray(surface_step.legal_action_mask, dtype=np.bool_)
    if action_mask_array.shape != (batch_size, player_count, 3):
        raise ValueError(
            "surface_step.legal_action_mask must have shape "
            f"({batch_size}, {player_count}, 3), got {action_mask_array.shape}"
        )
    action_mask = np.ascontiguousarray(action_mask_array[rows, players])

    reward_array = np.asarray(surface_step.reward, dtype=np.float32)
    if reward_array.shape != (batch_size, player_count):
        raise ValueError(
            f"surface_step.reward must have shape ({batch_size}, {player_count}), "
            f"got {reward_array.shape}"
        )
    reward = np.ascontiguousarray(reward_array[rows, players])

    done_by_row = np.asarray(surface_step.done, dtype=np.bool_)
    if done_by_row.shape != (batch_size,):
        raise ValueError(f"surface_step.done must have shape ({batch_size},), got {done_by_row.shape}")
    done = np.ascontiguousarray(done_by_row[rows])

    final_row_mask = np.asarray(
        getattr(surface_step, "final_observation_row_mask", np.zeros(batch_size, dtype=bool)),
        dtype=np.bool_,
    )
    if final_row_mask.shape != (batch_size,):
        raise ValueError(
            f"surface_step.final_observation_row_mask must have shape ({batch_size},), "
            f"got {final_row_mask.shape}"
        )
    final_observation = np.asarray(
        getattr(surface_step, "final_observation", np.zeros(expected_obs_shape, dtype=np.float32)),
        dtype=np.float32,
    )
    if final_observation.shape != expected_obs_shape:
        raise ValueError(
            f"surface_step.final_observation must have shape {expected_obs_shape}, "
            f"got {final_observation.shape}"
        )
    final_reward_map = np.asarray(
        getattr(surface_step, "final_reward_map", np.zeros((batch_size, player_count), dtype=np.float32)),
        dtype=np.float32,
    )
    if final_reward_map.shape != (batch_size, player_count):
        raise ValueError(
            "surface_step.final_reward_map must have shape "
            f"({batch_size}, {player_count}), got {final_reward_map.shape}"
        )

    info = []
    for row, player in zip(rows, players, strict=True):
        final_present = bool(final_row_mask[int(row)])
        item: dict[str, Any] = {
            "row": int(row),
            "player": int(player),
            "player_view": int(player),
            "profile_only": True,
            "surface_profile_loop": True,
            "scalar_env_id_timestep": True,
            "final_observation_present": final_present,
        }
        if final_present:
            item["final_observation"] = final_observation[int(row), int(player)].copy()
            item["final_reward"] = float(final_reward_map[int(row), int(player)])
        info.append(item)
    timestep = MockBaseEnvTimestep(
        obs={
            "observation": flat_obs,
            "action_mask": action_mask,
            "to_play": np.full((flat_obs.shape[0],), -1, dtype=np.int64),
        },
        reward=reward,
        done=done,
        info=info,
    )
    target_reward = reward.reshape(env_id_array.shape[0], 1).astype(np.float32, copy=False)
    return timestep, flat_obs, target_reward


def _scalarize_reward(
    reward: np.ndarray,
    *,
    batch_size: int,
    player_count: int,
) -> np.ndarray:
    array = np.asarray(reward, dtype=np.float32)
    if array.shape == (batch_size, player_count):
        return np.ascontiguousarray(array.reshape(batch_size * player_count))
    if array.shape == (batch_size,):
        return np.repeat(array, player_count).astype(np.float32, copy=False)
    if array.shape == (batch_size, 1):
        return np.repeat(array[:, 0], player_count).astype(np.float32, copy=False)
    raise ValueError(
        "step_reward must be shape "
        f"({batch_size}, {player_count}), ({batch_size},), or ({batch_size}, 1); "
        f"got {array.shape}"
    )


def _build_rnd_model(config: MockCollectorConfig) -> xb.CurvyRNDRewardModel:
    return xb.CurvyRNDRewardModel(
        {
            "input_type": "obs",
            "batch_size": int(config.rnd_batch_size),
            "update_per_collect": int(config.rnd_update_per_collect),
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
            "intrinsic_reward_weight": 0.0,
            "seed": int(config.seed),
            "curvyzero_adapter": {
                "shape": xb.RND_INPUT_SHAPE_POLICY_GRAY64_LATEST_V0,
                "source_observation_shape": xb.RND_SOURCE_OBSERVATION_SHAPE,
            },
        },
        device=str(config.rnd_device),
    )


def _elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def _ready_obs_by_env_id(
    *,
    env_ids: np.ndarray,
    timestep: MockBaseEnvTimestep,
) -> dict[int, dict[str, Any]]:
    observation = np.asarray(timestep.obs["observation"], dtype=np.float32)
    action_mask = np.asarray(timestep.obs["action_mask"], dtype=np.bool_)
    if observation.shape[0] != env_ids.shape[0] or action_mask.shape[0] != env_ids.shape[0]:
        raise ValueError("ready obs arrays must match env id count")
    return {
        int(env_id): {
            "observation": observation[index].copy(),
            "action_mask": action_mask[index].copy(),
            "to_play": -1,
        }
        for index, env_id in enumerate(env_ids)
    }


def _coerce_scalar_action(raw_action: Any) -> int:
    if isinstance(raw_action, Mapping):
        for key in ("action", "action_id"):
            if key in raw_action:
                return _coerce_scalar_action(raw_action[key])
        raise ValueError("action mapping must contain 'action' or 'action_id'")
    array = np.asarray(raw_action)
    if array.shape == ():
        return int(array.item())
    if array.size == 1:
        return int(array.reshape(-1)[0])
    raise ValueError(f"scalar action must contain one value, got shape {array.shape}")


def _split_timestep_by_env_id(
    *,
    env_ids: np.ndarray,
    timestep: MockBaseEnvTimestep,
) -> dict[int, MockBaseEnvTimestep]:
    observation = np.asarray(timestep.obs["observation"], dtype=np.float32)
    action_mask = np.asarray(timestep.obs["action_mask"], dtype=np.bool_)
    raw_to_play = timestep.obs.get("to_play", -1)
    to_play = np.asarray(raw_to_play, dtype=np.int64)
    if to_play.ndim == 0:
        to_play = np.full((env_ids.shape[0],), int(to_play), dtype=np.int64)
    reward = np.asarray(timestep.reward, dtype=np.float32)
    done = np.asarray(timestep.done, dtype=np.bool_)
    if observation.shape[0] != env_ids.shape[0]:
        raise ValueError("timestep observation rows must match env id count")
    if action_mask.shape[0] != env_ids.shape[0]:
        raise ValueError("timestep action masks must match env id count")
    if to_play.shape != (env_ids.shape[0],):
        raise ValueError("timestep to_play rows must match env id count")
    if reward.shape != (env_ids.shape[0],):
        raise ValueError("timestep reward rows must match env id count")
    if done.shape != (env_ids.shape[0],):
        raise ValueError("timestep done rows must match env id count")
    if len(timestep.info) != int(env_ids.shape[0]):
        raise ValueError("timestep info rows must match env id count")
    return {
        int(env_id): MockBaseEnvTimestep(
            obs={
                "observation": observation[index].copy(),
                "action_mask": action_mask[index].copy(),
                "to_play": int(to_play[index]),
            },
            reward=np.asarray(reward[index], dtype=np.float32),
            done=np.asarray(done[index], dtype=np.bool_),
            info=[dict(timestep.info[index])],
        )
        for index, env_id in enumerate(env_ids)
    }


def _timestep_materialization_counts(timestep: MockBaseEnvTimestep) -> dict[str, int]:
    observation = np.asarray(timestep.obs["observation"])
    action_mask = np.asarray(timestep.obs["action_mask"])
    reward = np.asarray(timestep.reward)
    done = np.asarray(timestep.done)
    final_observation_nbytes = 0
    for item in timestep.info:
        final_observation = item.get("final_observation")
        if final_observation is not None:
            final_observation_nbytes += int(np.asarray(final_observation).nbytes)
    return {
        "flat_obs_nbytes": int(observation.nbytes),
        "action_mask_nbytes": int(action_mask.nbytes),
        "reward_nbytes": int(reward.nbytes),
        "done_nbytes": int(done.nbytes),
        "final_observation_nbytes": int(final_observation_nbytes),
        "info_count": int(len(timestep.info)),
        "materialized_timestep_count": int(observation.shape[0]),
    }


def _to_stock_base_env_timestep(
    *,
    timestep: MockBaseEnvTimestep,
    base_env_timestep_cls: Any | None,
    reward: float | None = None,
    done: bool | None = None,
    info_overrides: Mapping[str, Any] | None = None,
) -> Any:
    if base_env_timestep_cls is None:
        return timestep
    info = dict(timestep.info[0]) if timestep.info else {}
    if info_overrides is not None:
        info.update(dict(info_overrides))
    reward_value = (
        float(np.asarray(timestep.reward, dtype=np.float32).reshape(()).item())
        if reward is None
        else float(reward)
    )
    done_value = (
        bool(np.asarray(timestep.done, dtype=np.bool_).reshape(()).item())
        if done is None
        else bool(done)
    )
    return base_env_timestep_cls(
        timestep.obs,
        reward_value,
        done_value,
        info,
    )


__all__ = [
    "BatchedLightZeroScalarActionBridge",
    "BatchedLightZeroProfileEnvManager",
    "BatchedLightZeroStockEnvManagerAdapter",
    "BatchedSourceStateTrainerProfileLoop",
    "BatchedSurfaceProfileLoopStep",
    "MockBaseEnvTimestep",
    "MockCollectorConfig",
    "ProfileEnvManagerStepResult",
    "SCHEMA_ID",
    "ScalarActionBridgeOutput",
    "materialize_lightzero_scalar_timestep",
    "materialize_trainer_surface_env_id_timestep",
    "materialize_trainer_surface_policy_timestep",
    "run_mock_collector_profile",
]
