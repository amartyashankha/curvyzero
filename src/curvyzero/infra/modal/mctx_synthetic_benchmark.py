"""Contained Modal GPU benchmark for synthetic Mctx Gumbel MuZero search.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
      --batch-size 8 \
      --num-simulations 4 \
      --steady-runs 2

    uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
      --observation-mode curvytron_debug \
      --batch-size 4 \
      --player-count 2 \
      --obs-dim 9 \
      --num-simulations 4 \
      --hidden-dim 32 \
      --max-depth 4 \
      --warmup-runs 1 \
      --steady-runs 2

    uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
      --observation-mode curvytron_debug_packer \
      --batch-size 4 \
      --player-count 2 \
      --obs-dim 9 \
      --num-simulations 4 \
      --hidden-dim 32 \
      --max-depth 4 \
      --warmup-runs 1 \
      --steady-runs 2

    uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
      --observation-mode curvytron_actor_bridge_sample \
      --batch-size 4 \
      --player-count 2 \
      --rollout-steps 2 \
      --num-simulations 4 \
      --hidden-dim 32 \
      --max-depth 4 \
      --warmup-runs 1 \
      --steady-runs 2

    uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
      --observation-mode curvytron_trainer_flat \
      --batch-size 64 \
      --player-count 2 \
      --obs-dim 106 \
      --num-simulations 8 \
      --hidden-dim 64 \
      --max-depth 8 \
      --warmup-runs 1 \
      --steady-runs 3

    uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
      --observation-mode curvytron_vector_trainer_sample \
      --batch-size 64 \
      --player-count 2 \
      --obs-dim 106 \
      --rollout-steps 2 \
      --num-simulations 8 \
      --hidden-dim 64 \
      --max-depth 8 \
      --warmup-runs 1 \
      --steady-runs 3

This is not a trainer. It measures one tiny fixed-shape synthetic search profile
on a cheap Modal GPU and prints clear JSON. The `curvytron_debug` mode is still
synthetic: it only mimics the current debug observation packer shape
`obs[B,P,9]`, then flattens ego rows into Mctx roots. The
`curvytron_debug_packer` mode builds `obs[B,P,9]` through the current
fixture-seeded CPU debug packer, filters live ego rows into Mctx roots, and then
times device-resident synthetic Mctx search separately. The
`curvytron_actor_bridge_sample` mode builds one fixed-shape sample through the
local vector actor bridge: fixture reset, real vector env step(s), debug
obs/reward packing, synthetic action feedback after step 0 when requested, then
live-ego filtering before synthetic Mctx search. The
`curvytron_vector_trainer_sample` mode builds one live sample from the strict
native `VectorTrainerEnv1v1NoBonus` observation contract `[B,2,106]`, maps live
legal policy rows, and then runs the same synthetic Mctx search timing. In all
CurvyTron-shaped modes, `--batch-size` is the env-row count.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
import statistics
import subprocess
import time
from importlib import metadata
from typing import Any

import modal

from curvyzero.infra.modal.mctx_dependency_smoke import JAX_VERSION
from curvyzero.infra.modal.mctx_dependency_smoke import MCTX_VERSION

APP_NAME = "curvyzero-mctx-synthetic-benchmark"
REMOTE_ROOT = Path("/repo")
ACTION_COUNT = 3
OBSERVATION_MODE_FLAT = "flat_hidden"
OBSERVATION_MODE_CURVYTRON_DEBUG = "curvytron_debug"
OBSERVATION_MODE_CURVYTRON_DEBUG_PACKER = "curvytron_debug_packer"
OBSERVATION_MODE_CURVYTRON_ACTOR_BRIDGE_SAMPLE = "curvytron_actor_bridge_sample"
OBSERVATION_MODE_CURVYTRON_TRAINER_FLAT = "curvytron_trainer_flat"
OBSERVATION_MODE_CURVYTRON_VECTOR_TRAINER_SAMPLE = "curvytron_vector_trainer_sample"
OBSERVATION_MODES = {
    OBSERVATION_MODE_FLAT,
    OBSERVATION_MODE_CURVYTRON_DEBUG,
    OBSERVATION_MODE_CURVYTRON_DEBUG_PACKER,
    OBSERVATION_MODE_CURVYTRON_ACTOR_BRIDGE_SAMPLE,
    OBSERVATION_MODE_CURVYTRON_TRAINER_FLAT,
    OBSERVATION_MODE_CURVYTRON_VECTOR_TRAINER_SAMPLE,
}
CURVYTRON_SHAPED_OBSERVATION_MODES = {
    OBSERVATION_MODE_CURVYTRON_DEBUG,
    OBSERVATION_MODE_CURVYTRON_DEBUG_PACKER,
    OBSERVATION_MODE_CURVYTRON_ACTOR_BRIDGE_SAMPLE,
    OBSERVATION_MODE_CURVYTRON_TRAINER_FLAT,
    OBSERVATION_MODE_CURVYTRON_VECTOR_TRAINER_SAMPLE,
}
DEBUG_OBS_FEATURE_NAMES = (
    "x_over_map_size",
    "y_over_map_size",
    "heading_sin",
    "heading_cos",
    "alive",
    "printing",
    "score",
    "round_score",
    "map_size_over_1000",
)
DEBUG_OBS_DIM = len(DEBUG_OBS_FEATURE_NAMES)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"mctx=={MCTX_VERSION}",
        f"jax[cuda12]=={JAX_VERSION}",
        "numpy>=1.26",
    )
    .env({"PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'scripts'}"})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(
        Path.cwd() / "scripts",
        remote_path=str(REMOTE_ROOT / "scripts"),
        copy=True,
    )
    .add_local_dir(
        Path.cwd() / "scenarios",
        remote_path=str(REMOTE_ROOT / "scenarios"),
        copy=True,
    )
)

app = modal.App(APP_NAME)


def _version_or_missing(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def _nvidia_smi() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _positive_int(config: dict[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")
    return value


def _nonnegative_int(config: dict[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def _positive_float(config: dict[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be > 0, got {value}")
    return value


def _optional_positive_int(config: dict[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, 0))
    if value == 0:
        return default
    if value < 1:
        raise ValueError(f"{key} must be 0 or >= 1, got {value}")
    return value


def _optional_fixture_paths(config: dict[str, Any]) -> list[str] | None:
    raw_value = config.get("fixture_paths")
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        paths = [part.strip() for part in raw_value.split(",") if part.strip()]
        return paths or None
    if isinstance(raw_value, (list, tuple)):
        paths = [str(part) for part in raw_value if str(part).strip()]
        return paths or None
    raise ValueError("fixture_paths must be a comma-separated string or list")


def _compact_packer_source(source_summary: dict[str, Any]) -> dict[str, Any]:
    selected_group = dict(source_summary["selected_group"])
    preflight = dict(selected_group["preflight"])
    selected_group["preflight"] = {
        "row_count": preflight["row_count"],
        "match": preflight["match"],
        "state_match": preflight["state_match"],
        "event_match": preflight["event_match"],
        "mismatch_count": len(preflight["mismatches"]),
        "batch_counters": preflight["batch_counters"],
        "expected_scalar_counters": preflight["expected_scalar_counters"],
    }
    return {
        "schema": source_summary["schema"],
        "benchmark_id": source_summary["benchmark_id"],
        "source": source_summary["source"],
        "source_fidelity_claim": source_summary["source_fidelity_claim"],
        "trust_level": source_summary["trust_level"],
        "config": source_summary["config"],
        "summary": source_summary["summary"],
        "input_count": source_summary["input_count"],
        "fixture_count": source_summary["fixture_count"],
        "supported_fixture_count": source_summary["supported_fixture_count"],
        "selected_group": selected_group,
        "step_counters": source_summary["step_counters"],
        "sample": source_summary["sample"],
        "timing_sec": source_summary["timing_sec"],
        "known_fake_or_incomplete": source_summary["known_fake_or_incomplete"],
    }


def _compact_actor_bridge_source(source_summary: dict[str, Any]) -> dict[str, Any]:
    selected_group = dict(source_summary["selected_group"])
    preflight = dict(selected_group["preflight"])
    selected_group["preflight"] = {
        "row_count": preflight["row_count"],
        "match": preflight["match"],
        "state_match": preflight["state_match"],
        "event_match": preflight["event_match"],
        "mismatch_count": len(preflight["mismatches"]),
        "batch_counters": preflight["batch_counters"],
        "expected_scalar_counters": preflight["expected_scalar_counters"],
    }
    no_event_preflight = selected_group.get("no_event_preflight")
    if no_event_preflight is not None:
        no_event_preflight = dict(no_event_preflight)
        no_event_mismatches = [
            *no_event_preflight["state_mismatches"],
            *no_event_preflight["counter_mismatches"],
        ]
        selected_group["no_event_preflight"] = {
            "row_count": no_event_preflight["row_count"],
            "match": no_event_preflight["match"],
            "state_match": no_event_preflight["state_match"],
            "mismatch_count": len(no_event_mismatches),
            "batch_counters": no_event_preflight["batch_counters"],
            "expected_scalar_counters": no_event_preflight["expected_scalar_counters"],
        }
    return {
        "schema": source_summary["schema"],
        "benchmark_id": source_summary["benchmark_id"],
        "source": source_summary["source"],
        "source_fidelity_claim": source_summary["source_fidelity_claim"],
        "trust_level": source_summary["trust_level"],
        "config": source_summary["config"],
        "summary": source_summary["summary"],
        "input_count": source_summary["input_count"],
        "fixture_count": source_summary["fixture_count"],
        "supported_fixture_count": source_summary["supported_fixture_count"],
        "selected_group": selected_group,
        "step_counters": source_summary["step_counters"],
        "step_summaries": source_summary["step_summaries"],
        "sample": source_summary["sample"],
        "timing_sec": source_summary["timing_sec"],
        "known_fake_or_incomplete": source_summary["known_fake_or_incomplete"],
    }


def _run_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import mctx
    import numpy as np

    batch_size = _positive_int(config, "batch_size", 16)
    num_simulations = _positive_int(config, "num_simulations", 8)
    hidden_dim = _positive_int(config, "hidden_dim", 32)
    max_depth = _positive_int(config, "max_depth", 8)
    warmup_runs = _nonnegative_int(config, "warmup_runs", 1)
    steady_runs = _positive_int(config, "steady_runs", 5)
    player_count = _positive_int(config, "player_count", 2)
    observation_mode = str(config.get("observation_mode", OBSERVATION_MODE_FLAT))
    if observation_mode not in OBSERVATION_MODES:
        allowed = ", ".join(sorted(OBSERVATION_MODES))
        raise ValueError(f"observation_mode must be one of {allowed}, got {observation_mode!r}")

    devices = jax.devices()
    backend = jax.default_backend()
    problems: list[str] = []
    if backend not in {"gpu", "cuda"}:
        problems.append(f"expected a GPU JAX backend, got {backend!r}")

    def linspace_matrix(rows: int, cols: int, scale: float) -> Any:
        return jnp.linspace(-scale, scale, rows * cols, dtype=jnp.float32).reshape(
            rows, cols
        )

    candidate_ego_rows = batch_size
    if observation_mode == OBSERVATION_MODE_CURVYTRON_DEBUG:
        obs_dim = _optional_positive_int(config, "obs_dim", DEBUG_OBS_DIM)
        if obs_dim != DEBUG_OBS_DIM:
            raise ValueError(
                f"{OBSERVATION_MODE_CURVYTRON_DEBUG} requires obs_dim={DEBUG_OBS_DIM}, "
                f"got {obs_dim}"
            )
        env_batch_size = batch_size
        candidate_ego_rows = env_batch_size * player_count
        root_batch_size = candidate_ego_rows
        host_setup_started = time.perf_counter()
        env_index = np.arange(env_batch_size, dtype=np.float32)[:, None]
        player_index = np.arange(player_count, dtype=np.float32)[None, :]
        heading = 0.37 * env_index + 1.13 * player_index
        obs_env_host = np.stack(
            [
                (env_index + 1.0 + 0.125 * player_index)
                / (env_batch_size + 2.0),
                (0.5 * env_index + player_index + 1.0)
                / (env_batch_size + player_count + 1.0),
                np.sin(heading),
                np.cos(heading),
                np.ones((env_batch_size, player_count), dtype=np.float32),
                np.mod(env_index + player_index, 2.0),
                np.mod(env_index + player_index, 5.0),
                np.mod(2.0 * env_index + player_index, 3.0),
                np.full((env_batch_size, player_count), 0.8, dtype=np.float32),
            ],
            axis=-1,
        ).astype(np.float32)
        ego_mask_host = np.ones((env_batch_size, player_count), dtype=np.bool_)
        legal_action_mask_host = np.broadcast_to(
            ego_mask_host[:, :, None],
            (env_batch_size, player_count, ACTION_COUNT),
        ).copy()
        ego_row_id_host = np.arange(root_batch_size, dtype=np.int32).reshape(
            env_batch_size,
            player_count,
        )
        ego_env_id_host = np.broadcast_to(
            np.arange(env_batch_size, dtype=np.int32)[:, None],
            (env_batch_size, player_count),
        ).copy()
        ego_player_id_host = np.broadcast_to(
            np.arange(player_count, dtype=np.int32)[None, :],
            (env_batch_size, player_count),
        ).copy()
        obs_host = obs_env_host.reshape(root_batch_size, obs_dim)
        invalid_actions_host = ~legal_action_mask_host.reshape(
            root_batch_size, ACTION_COUNT
        )
        live_ego_rows = int(ego_mask_host.sum())
        expected_source_shape = (env_batch_size, player_count, DEBUG_OBS_DIM)
        expected_root_shape = (candidate_ego_rows, DEBUG_OBS_DIM)
        if obs_env_host.shape != expected_source_shape:
            raise AssertionError(
                f"expected obs_env shape {expected_source_shape}, got {obs_env_host.shape}"
            )
        if obs_host.shape != expected_root_shape:
            raise AssertionError(
                f"expected flattened obs shape {expected_root_shape}, got {obs_host.shape}"
            )
        if invalid_actions_host.shape != (root_batch_size, ACTION_COUNT):
            raise AssertionError(
                "expected invalid action mask shape "
                f"{(root_batch_size, ACTION_COUNT)}, got {invalid_actions_host.shape}"
            )
        if live_ego_rows != root_batch_size:
            raise AssertionError(
                f"expected {candidate_ego_rows} live ego rows, got {live_ego_rows}"
            )
        host_observation_setup_sec = time.perf_counter() - host_setup_started
        transfer_started = time.perf_counter()
        obs = jax.device_put(obs_host)
        invalid_actions = jax.device_put(invalid_actions_host)
        obs.block_until_ready()
        invalid_actions.block_until_ready()
        host_to_device_transfer_sec = time.perf_counter() - transfer_started
        observation = {
            "mode": observation_mode,
            "source": "synthetic_curvytron_debug_obs_shape",
            "build_path": "host_numpy_then_jax_device_put",
            "source_tensor_shape": list(obs_env_host.shape),
            "root_obs_shape": list(obs_host.shape),
            "obs_features": list(DEBUG_OBS_FEATURE_NAMES),
            "ego_mask_shape": list(ego_mask_host.shape),
            "legal_action_mask_shape": list(legal_action_mask_host.shape),
            "ego_row_id_shape": list(ego_row_id_host.shape),
            "ego_env_id_shape": list(ego_env_id_host.shape),
            "ego_player_id_shape": list(ego_player_id_host.shape),
            "candidate_ego_rows": candidate_ego_rows,
            "live_ego_rows": live_ego_rows,
            "search_root_count": root_batch_size,
            "shape_assertions": "passed",
            "note": (
                "Synthetic shape-only stand-in for the debug packer. It is not "
                "CurvyTron rollout throughput, replay, reward learning, or "
                "source-fidelity evidence."
            ),
        }
    elif observation_mode == OBSERVATION_MODE_CURVYTRON_DEBUG_PACKER:
        obs_dim = _optional_positive_int(config, "obs_dim", DEBUG_OBS_DIM)
        if obs_dim != DEBUG_OBS_DIM:
            raise ValueError(
                f"{OBSERVATION_MODE_CURVYTRON_DEBUG_PACKER} requires "
                f"obs_dim={DEBUG_OBS_DIM}, got {obs_dim}"
            )

        from benchmark_vector_obs_reward_packing import (
            DEBUG_OBS_SCHEMA,
            DEBUG_REWARD_SCHEMA,
            build_fixture_seeded_debug_surfaces,
        )

        host_setup_started = time.perf_counter()
        fixture_paths = _optional_fixture_paths(config)
        surface_kwargs = {
            "body_capacity": _nonnegative_int(config, "body_capacity", 4),
            "step_index": _nonnegative_int(config, "step_index", 0),
            "batch_size": batch_size,
            "player_count": player_count,
            "require_verified": not bool(config.get("allow_unverified", False)),
        }
        if fixture_paths is None:
            surface_payload = build_fixture_seeded_debug_surfaces(**surface_kwargs)
        else:
            surface_payload = build_fixture_seeded_debug_surfaces(
                fixture_paths,
                **surface_kwargs,
            )
        surfaces = surface_payload["surfaces"]
        source_summary = surface_payload["source"]

        obs_env_host = np.asarray(surfaces["obs"], dtype=np.float32)
        reward_host = np.asarray(surfaces["reward"], dtype=np.float32)
        done_host = np.asarray(surfaces["done"], dtype=np.bool_)
        truncated_host = np.asarray(surfaces["truncated"], dtype=np.bool_)
        ego_mask_host = np.asarray(surfaces["ego_mask"], dtype=np.bool_)
        legal_action_mask_host = np.asarray(surfaces["legal_action_mask"], dtype=np.bool_)
        ego_row_id_host = np.asarray(surfaces["ego_row_id"], dtype=np.int32)
        ego_env_id_host = np.asarray(surfaces["ego_env_id"], dtype=np.int32)
        ego_player_id_host = np.asarray(surfaces["ego_player_id"], dtype=np.int16)

        env_batch_size = int(obs_env_host.shape[0])
        player_count = int(obs_env_host.shape[1])
        candidate_ego_rows = env_batch_size * player_count
        live_flat_host = ego_mask_host.reshape(candidate_ego_rows)
        live_ego_rows = int(live_flat_host.sum())
        if live_ego_rows < 1:
            raise ValueError("fixture debug packer produced no live ego rows")

        expected_source_shape = (env_batch_size, player_count, DEBUG_OBS_DIM)
        expected_legal_shape = (env_batch_size, player_count, ACTION_COUNT)
        if obs_env_host.shape != expected_source_shape:
            raise AssertionError(
                f"expected obs_env shape {expected_source_shape}, got {obs_env_host.shape}"
            )
        if legal_action_mask_host.shape != expected_legal_shape:
            raise AssertionError(
                "expected legal action mask shape "
                f"{expected_legal_shape}, got {legal_action_mask_host.shape}"
            )
        if ego_mask_host.shape != (env_batch_size, player_count):
            raise AssertionError(
                "expected ego_mask shape "
                f"{(env_batch_size, player_count)}, got {ego_mask_host.shape}"
            )

        obs_flat_host = obs_env_host.reshape(candidate_ego_rows, obs_dim)
        legal_flat_host = legal_action_mask_host.reshape(
            candidate_ego_rows,
            ACTION_COUNT,
        )
        obs_host = obs_flat_host[live_flat_host]
        invalid_actions_host = ~legal_flat_host[live_flat_host]
        if not np.any(~invalid_actions_host, axis=1).all():
            raise AssertionError("every live Mctx root must have at least one legal action")
        root_batch_size = live_ego_rows
        host_observation_setup_sec = time.perf_counter() - host_setup_started

        transfer_started = time.perf_counter()
        obs = jax.device_put(obs_host)
        invalid_actions = jax.device_put(invalid_actions_host)
        obs.block_until_ready()
        invalid_actions.block_until_ready()
        host_to_device_transfer_sec = time.perf_counter() - transfer_started

        reward_components = surfaces["reward_components"]
        source_sample = source_summary["sample"]
        observation = {
            "mode": observation_mode,
            "source": "fixture_seeded_cpu_debug_packer_output",
            "build_path": (
                "seed_fixtures_source_preflight_batched_step_pack_debug_obs_reward_"
                "then_live_ego_filter_then_jax_device_put"
            ),
            "obs_schema": DEBUG_OBS_SCHEMA,
            "reward_schema": DEBUG_REWARD_SCHEMA,
            "source_tensor_shape": list(obs_env_host.shape),
            "root_obs_shape": list(obs_host.shape),
            "obs_features": list(DEBUG_OBS_FEATURE_NAMES),
            "reward_shape": list(reward_host.shape),
            "done_shape": list(done_host.shape),
            "truncated_shape": list(truncated_host.shape),
            "ego_mask_shape": list(ego_mask_host.shape),
            "legal_action_mask_shape": list(legal_action_mask_host.shape),
            "ego_row_id_shape": list(ego_row_id_host.shape),
            "ego_env_id_shape": list(ego_env_id_host.shape),
            "ego_player_id_shape": list(ego_player_id_host.shape),
            "candidate_ego_rows": candidate_ego_rows,
            "live_ego_rows": live_ego_rows,
            "search_root_count": root_batch_size,
            "root_filter": "ego_mask live rows only",
            "root_ego_row_id_sample": ego_row_id_host.reshape(candidate_ego_rows)[
                live_flat_host
            ][: min(root_batch_size, 8)].astype(int).tolist(),
            "root_ego_env_id_sample": ego_env_id_host.reshape(candidate_ego_rows)[
                live_flat_host
            ][: min(root_batch_size, 8)].astype(int).tolist(),
            "root_ego_player_id_sample": ego_player_id_host.reshape(candidate_ego_rows)[
                live_flat_host
            ][: min(root_batch_size, 8)].astype(int).tolist(),
            "reward_sum": float(reward_host.sum(dtype=np.float64)),
            "reward_died_source": str(reward_components["died_source"]),
            "surface_checksum": source_sample["checksum"],
            "surface_bytes": source_sample["bytes_per_pack"],
            "packer_source": _compact_packer_source(source_summary),
            "shape_assertions": "passed",
            "note": (
                "Uses the current fixture-seeded CPU debug packer output, then "
                "runs synthetic Mctx search over live ego rows. It is still not "
                "a real rollout, learned dynamics, replay, trainer, or final "
                "training observation/reward contract."
            ),
        }
    elif observation_mode == OBSERVATION_MODE_CURVYTRON_ACTOR_BRIDGE_SAMPLE:
        obs_dim = _optional_positive_int(config, "obs_dim", DEBUG_OBS_DIM)
        if obs_dim != DEBUG_OBS_DIM:
            raise ValueError(
                f"{OBSERVATION_MODE_CURVYTRON_ACTOR_BRIDGE_SAMPLE} requires "
                f"obs_dim={DEBUG_OBS_DIM}, got {obs_dim}"
            )

        from benchmark_vector_actor_loop_bridge import (
            build_fixture_seeded_actor_bridge_sample,
        )
        from benchmark_vector_obs_reward_packing import (
            DEBUG_OBS_SCHEMA,
            DEBUG_REWARD_SCHEMA,
        )

        host_setup_started = time.perf_counter()
        fixture_paths = _optional_fixture_paths(config)
        sample_kwargs = {
            "body_capacity": _nonnegative_int(config, "body_capacity", 4),
            "step_index": _nonnegative_int(config, "step_index", 0),
            "batch_size": batch_size,
            "player_count": player_count,
            "event_mode": str(config.get("event_mode", "debug-event")),
            "rollout_steps": _positive_int(config, "rollout_steps", 2),
            "hidden_dim": hidden_dim,
            "simulations": _positive_int(config, "actor_simulations", 4),
            "seed": int(config.get("actor_seed", 0)),
            "require_verified": not bool(config.get("allow_unverified", False)),
        }
        group_id = str(config.get("group_id", "")).strip()
        if group_id:
            sample_kwargs["group_id"] = group_id
        if fixture_paths is None:
            sample_payload = build_fixture_seeded_actor_bridge_sample(**sample_kwargs)
        else:
            sample_payload = build_fixture_seeded_actor_bridge_sample(
                fixture_paths,
                **sample_kwargs,
            )
        surfaces = sample_payload["surfaces"]
        source_summary = sample_payload["source"]

        obs_env_host = np.asarray(surfaces["obs"], dtype=np.float32)
        reward_host = np.asarray(surfaces["reward"], dtype=np.float32)
        done_host = np.asarray(surfaces["done"], dtype=np.bool_)
        truncated_host = np.asarray(surfaces["truncated"], dtype=np.bool_)
        ego_mask_host = np.asarray(surfaces["ego_mask"], dtype=np.bool_)
        legal_action_mask_host = np.asarray(surfaces["legal_action_mask"], dtype=np.bool_)
        ego_row_id_host = np.asarray(surfaces["ego_row_id"], dtype=np.int32)
        ego_env_id_host = np.asarray(surfaces["ego_env_id"], dtype=np.int32)
        ego_player_id_host = np.asarray(surfaces["ego_player_id"], dtype=np.int16)

        env_batch_size = int(obs_env_host.shape[0])
        player_count = int(obs_env_host.shape[1])
        candidate_ego_rows = env_batch_size * player_count
        live_flat_host = ego_mask_host.reshape(candidate_ego_rows)
        live_ego_rows = int(live_flat_host.sum())
        if live_ego_rows < 1:
            raise ValueError("actor bridge sample produced no live ego rows")

        expected_source_shape = (env_batch_size, player_count, DEBUG_OBS_DIM)
        expected_legal_shape = (env_batch_size, player_count, ACTION_COUNT)
        if obs_env_host.shape != expected_source_shape:
            raise AssertionError(
                f"expected obs_env shape {expected_source_shape}, got {obs_env_host.shape}"
            )
        if legal_action_mask_host.shape != expected_legal_shape:
            raise AssertionError(
                "expected legal action mask shape "
                f"{expected_legal_shape}, got {legal_action_mask_host.shape}"
            )
        if ego_mask_host.shape != (env_batch_size, player_count):
            raise AssertionError(
                "expected ego_mask shape "
                f"{(env_batch_size, player_count)}, got {ego_mask_host.shape}"
            )

        obs_flat_host = obs_env_host.reshape(candidate_ego_rows, obs_dim)
        legal_flat_host = legal_action_mask_host.reshape(
            candidate_ego_rows,
            ACTION_COUNT,
        )
        obs_host = obs_flat_host[live_flat_host]
        invalid_actions_host = ~legal_flat_host[live_flat_host]
        if not np.any(~invalid_actions_host, axis=1).all():
            raise AssertionError("every live Mctx root must have at least one legal action")
        root_batch_size = live_ego_rows
        host_observation_setup_sec = time.perf_counter() - host_setup_started

        transfer_started = time.perf_counter()
        obs = jax.device_put(obs_host)
        invalid_actions = jax.device_put(invalid_actions_host)
        obs.block_until_ready()
        invalid_actions.block_until_ready()
        host_to_device_transfer_sec = time.perf_counter() - transfer_started

        reward_components = surfaces["reward_components"]
        source_sample = source_summary["sample"]
        step_summaries = source_summary["step_summaries"]
        final_step = step_summaries[-1]
        observation = {
            "mode": observation_mode,
            "source": "fixture_seeded_cpu_actor_loop_bridge_sample",
            "build_path": (
                "seed_fixtures_source_preflight_batched_step_actor_bridge_sample_"
                "then_live_ego_filter_then_jax_device_put"
            ),
            "obs_schema": DEBUG_OBS_SCHEMA,
            "reward_schema": DEBUG_REWARD_SCHEMA,
            "source_tensor_shape": list(obs_env_host.shape),
            "root_obs_shape": list(obs_host.shape),
            "obs_features": list(DEBUG_OBS_FEATURE_NAMES),
            "reward_shape": list(reward_host.shape),
            "done_shape": list(done_host.shape),
            "truncated_shape": list(truncated_host.shape),
            "ego_mask_shape": list(ego_mask_host.shape),
            "legal_action_mask_shape": list(legal_action_mask_host.shape),
            "ego_row_id_shape": list(ego_row_id_host.shape),
            "ego_env_id_shape": list(ego_env_id_host.shape),
            "ego_player_id_shape": list(ego_player_id_host.shape),
            "candidate_ego_rows": candidate_ego_rows,
            "live_ego_rows": live_ego_rows,
            "search_root_count": root_batch_size,
            "root_filter": "ego_mask live rows only",
            "rollout_steps": sample_kwargs["rollout_steps"],
            "event_mode": sample_kwargs["event_mode"],
            "actor_simulations": sample_kwargs["simulations"],
            "final_step_source_kind": final_step["source_kind"],
            "root_ego_row_id_sample": ego_row_id_host.reshape(candidate_ego_rows)[
                live_flat_host
            ][: min(root_batch_size, 8)].astype(int).tolist(),
            "root_ego_env_id_sample": ego_env_id_host.reshape(candidate_ego_rows)[
                live_flat_host
            ][: min(root_batch_size, 8)].astype(int).tolist(),
            "root_ego_player_id_sample": ego_player_id_host.reshape(candidate_ego_rows)[
                live_flat_host
            ][: min(root_batch_size, 8)].astype(int).tolist(),
            "reward_sum": float(reward_host.sum(dtype=np.float64)),
            "reward_died_source": str(reward_components["died_source"]),
            "surface_checksum": source_sample["checksum"],
            "surface_bytes": source_sample["bytes_per_sample"],
            "actor_bridge_source": _compact_actor_bridge_source(source_summary),
            "shape_assertions": "passed",
            "note": (
                "Uses one fixed-shape sample from the current fixture-reset actor "
                "bridge. The vector env steps are real NumPy vector steps and the "
                "final obs/reward/legal masks are real debug packer output. It is "
                "still fixture-seeded, uses synthetic feedback actions after step "
                "0, and runs synthetic Mctx search."
            ),
        }
    elif observation_mode == OBSERVATION_MODE_CURVYTRON_TRAINER_FLAT:
        obs_dim = _optional_positive_int(config, "obs_dim", 106)
        env_batch_size = batch_size
        candidate_ego_rows = env_batch_size * player_count
        root_batch_size = candidate_ego_rows
        host_setup_started = time.perf_counter()
        env_index = np.arange(env_batch_size, dtype=np.float32)[:, None, None]
        player_index = np.arange(player_count, dtype=np.float32)[None, :, None]
        feature_index = np.arange(obs_dim, dtype=np.float32)[None, None, :]
        obs_env_host = np.sin(
            0.013 * feature_index + 0.071 * env_index + 0.19 * player_index
        ).astype(np.float32)
        ego_mask_host = np.ones((env_batch_size, player_count), dtype=np.bool_)
        legal_action_mask_host = np.broadcast_to(
            ego_mask_host[:, :, None],
            (env_batch_size, player_count, ACTION_COUNT),
        ).copy()
        ego_row_id_host = np.arange(candidate_ego_rows, dtype=np.int32).reshape(
            env_batch_size,
            player_count,
        )
        ego_env_id_host = np.broadcast_to(
            np.arange(env_batch_size, dtype=np.int32)[:, None],
            (env_batch_size, player_count),
        ).copy()
        ego_player_id_host = np.broadcast_to(
            np.arange(player_count, dtype=np.int32)[None, :],
            (env_batch_size, player_count),
        ).copy()
        obs_host = obs_env_host.reshape(root_batch_size, obs_dim)
        invalid_actions_host = ~legal_action_mask_host.reshape(
            root_batch_size,
            ACTION_COUNT,
        )
        live_ego_rows = int(ego_mask_host.sum())
        host_observation_setup_sec = time.perf_counter() - host_setup_started

        transfer_started = time.perf_counter()
        obs = jax.device_put(obs_host)
        invalid_actions = jax.device_put(invalid_actions_host)
        obs.block_until_ready()
        invalid_actions.block_until_ready()
        host_to_device_transfer_sec = time.perf_counter() - transfer_started

        observation = {
            "mode": observation_mode,
            "source": "synthetic_curvytron_trainer_flat_obs_shape",
            "build_path": "host_numpy_then_jax_device_put",
            "source_tensor_shape": list(obs_env_host.shape),
            "root_obs_shape": list(obs_host.shape),
            "ego_mask_shape": list(ego_mask_host.shape),
            "legal_action_mask_shape": list(legal_action_mask_host.shape),
            "ego_row_id_shape": list(ego_row_id_host.shape),
            "ego_env_id_shape": list(ego_env_id_host.shape),
            "ego_player_id_shape": list(ego_player_id_host.shape),
            "candidate_ego_rows": candidate_ego_rows,
            "live_ego_rows": live_ego_rows,
            "search_root_count": root_batch_size,
            "shape_assertions": "passed",
            "note": (
                "Synthetic stand-in for trainer flat observations shaped "
                "[B,P,106]. It measures host tensor setup, H2D transfer, and "
                "device-resident synthetic Mctx search, not CPU ray generation "
                "or CurvyTron source-fidelity."
            ),
        }
    elif observation_mode == OBSERVATION_MODE_CURVYTRON_VECTOR_TRAINER_SAMPLE:
        from curvyzero.env.trainer_contract import ACTION_SPACE_ID
        from curvyzero.env.trainer_contract import LIGHTZERO_FLAT_OBSERVATION_SHAPE
        from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_HASH
        from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_ID
        from curvyzero.env.trainer_contract import RAY_ANGLES_DEGREES
        from curvyzero.env.trainer_contract import RAY_CHANNEL_NAMES
        from curvyzero.env.trainer_contract import SCALAR_NAMES
        from curvyzero.env.vector_trainer_env import VectorTrainerEnv1v1NoBonus
        from curvyzero.training.policy_row_mapping import build_policy_row_mapping

        if player_count != 2:
            raise ValueError(
                f"{OBSERVATION_MODE_CURVYTRON_VECTOR_TRAINER_SAMPLE} requires "
                f"player_count=2, got {player_count}"
            )
        obs_dim = _optional_positive_int(config, "obs_dim", LIGHTZERO_FLAT_OBSERVATION_SHAPE[0])
        if obs_dim != LIGHTZERO_FLAT_OBSERVATION_SHAPE[0]:
            raise ValueError(
                f"{OBSERVATION_MODE_CURVYTRON_VECTOR_TRAINER_SAMPLE} requires "
                f"obs_dim={LIGHTZERO_FLAT_OBSERVATION_SHAPE[0]}, got {obs_dim}"
            )

        env_batch_size = batch_size
        candidate_ego_rows = env_batch_size * player_count
        seed = _nonnegative_int(config, "seed", 123)
        decision_ms = _positive_float(config, "decision_ms", 1000.0 / 60.0)
        event_mode = str(config.get("event_mode") or "no-event")
        rollout_steps = _nonnegative_int(config, "rollout_steps", 0)
        row_seeds = seed + np.arange(env_batch_size, dtype=np.int64)

        host_setup_started = time.perf_counter()
        env_init_started = time.perf_counter()
        env = VectorTrainerEnv1v1NoBonus(
            batch_size=env_batch_size,
            seed=seed,
            decision_ms=decision_ms,
            event_mode=event_mode,
        )
        env_init_sec = time.perf_counter() - env_init_started

        reset_started = time.perf_counter()
        batch = env.reset(seed=row_seeds)
        env_reset_sec = time.perf_counter() - reset_started

        rollout_step_times_sec: list[float] = []
        rollout_terminal_counts: list[int] = []
        for _ in range(rollout_steps):
            legal = np.asarray(batch.action_mask, dtype=np.bool_)
            actions = np.full((env_batch_size, player_count), 1, dtype=np.int8)
            legal_any = legal.any(axis=2)
            straight_illegal = ~legal[:, :, 1]
            fallback_actions = np.argmax(legal, axis=2).astype(np.int8, copy=False)
            fallback_mask = straight_illegal & legal_any
            actions[fallback_mask] = fallback_actions[fallback_mask]

            step_started = time.perf_counter()
            batch = env.step(actions)
            rollout_step_times_sec.append(time.perf_counter() - step_started)
            rollout_terminal_counts.append(int(np.asarray(batch.done, dtype=bool).sum()))

        mapping_started = time.perf_counter()
        obs_env_host = np.asarray(batch.observation, dtype=np.float32)
        legal_action_mask_host = np.asarray(batch.action_mask, dtype=np.bool_)
        lightzero_action_mask_host = np.asarray(batch.lightzero_action_mask, dtype=np.int8)
        live_mask_host = legal_action_mask_host.any(axis=2)
        mapping = build_policy_row_mapping(
            obs_env_host,
            live_mask_host,
            legal_action_mask_host,
            pad_to=candidate_ego_rows,
        )
        obs_host = np.asarray(mapping.observations, dtype=np.float32)
        invalid_actions_host = ~np.asarray(mapping.legal_action_mask, dtype=np.bool_)
        mapping_setup_sec = time.perf_counter() - mapping_started

        expected_source_shape = (
            env_batch_size,
            player_count,
            LIGHTZERO_FLAT_OBSERVATION_SHAPE[0],
        )
        expected_action_shape = (env_batch_size, player_count, ACTION_COUNT)
        if obs_env_host.shape != expected_source_shape:
            raise AssertionError(
                f"expected obs_env shape {expected_source_shape}, got {obs_env_host.shape}"
            )
        if legal_action_mask_host.shape != expected_action_shape:
            raise AssertionError(
                "expected action mask shape "
                f"{expected_action_shape}, got {legal_action_mask_host.shape}"
            )
        if obs_host.shape != (candidate_ego_rows, obs_dim):
            raise AssertionError(
                "expected mapped obs shape "
                f"{(candidate_ego_rows, obs_dim)}, got {obs_host.shape}"
            )
        if invalid_actions_host.shape != (candidate_ego_rows, ACTION_COUNT):
            raise AssertionError(
                "expected invalid action mask shape "
                f"{(candidate_ego_rows, ACTION_COUNT)}, got {invalid_actions_host.shape}"
            )
        if mapping.active_count < 1:
            raise ValueError("vector trainer sample produced no live legal policy rows")

        root_batch_size = mapping.capacity
        live_ego_rows = mapping.active_count
        host_observation_setup_sec = time.perf_counter() - host_setup_started

        transfer_started = time.perf_counter()
        obs = jax.device_put(obs_host)
        invalid_actions = jax.device_put(invalid_actions_host)
        obs.block_until_ready()
        invalid_actions.block_until_ready()
        host_to_device_transfer_sec = time.perf_counter() - transfer_started

        active_rows = np.asarray(mapping.row_mask, dtype=bool)
        observation = {
            "mode": observation_mode,
            "source": "VectorTrainerEnv1v1NoBonus",
            "build_path": (
                "VectorTrainerEnv1v1NoBonus_reset_with_row_seeds_optional_"
                "straight_rollout_build_policy_row_mapping_padded_to_BxP_"
                "then_jax_device_put"
            ),
            "source_tensor_shape": list(obs_env_host.shape),
            "root_obs_shape": list(obs_host.shape),
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
            "action_space_id": ACTION_SPACE_ID,
            "obs_features": {
                "rays": {
                    "angles_degrees": list(RAY_ANGLES_DEGREES),
                    "channels": list(RAY_CHANNEL_NAMES),
                },
                "scalars": list(SCALAR_NAMES),
            },
            "feature_flags": ["strict_1v1", "no_bonus", "P=2"],
            "caveat": "Strict VectorTrainerEnv1v1NoBonus 1v1/no_bonus sample only.",
            "seed": seed,
            "row_seed_sample": row_seeds[: min(env_batch_size, 8)].astype(int).tolist(),
            "decision_ms": decision_ms,
            "event_mode": event_mode,
            "rollout_steps": rollout_steps,
            "rollout_terminal_counts": rollout_terminal_counts,
            "env_body_capacity": int(env.body_capacity),
            "env_event_capacity": int(env.event_capacity),
            "env_random_tape_capacity": int(env.random_tape_capacity),
            "live_mask_shape": list(live_mask_host.shape),
            "source_action_mask_shape": list(legal_action_mask_host.shape),
            "lightzero_action_mask_shape": list(lightzero_action_mask_host.shape),
            "mapped_legal_action_mask_shape": list(mapping.legal_action_mask.shape),
            "invalid_action_mask_shape": list(invalid_actions_host.shape),
            "candidate_ego_rows": candidate_ego_rows,
            "live_ego_rows": live_ego_rows,
            "padded_policy_rows": int(root_batch_size - live_ego_rows),
            "search_root_count": root_batch_size,
            "root_filter": (
                "build_policy_row_mapping live rows where batch.action_mask has at "
                "least one legal action; padded to B*P with row_mask=false"
            ),
            "mapping_schema": mapping.schema,
            "mapping_source_shape": list(mapping.source_shape),
            "mapping_capacity": mapping.capacity,
            "mapping_active_count": mapping.active_count,
            "mapping_row_mask_shape": list(mapping.row_mask.shape),
            "root_env_id_sample": mapping.env_row_id[active_rows][
                : min(live_ego_rows, 8)
            ].astype(int).tolist(),
            "root_player_id_sample": mapping.player_id[active_rows][
                : min(live_ego_rows, 8)
            ].astype(int).tolist(),
            "env_step_setup_timing_sec": {
                "env_init_sec": env_init_sec,
                "env_reset_sec": env_reset_sec,
                "env_rollout_step_times_sec": rollout_step_times_sec,
                "env_rollout_total_sec": float(sum(rollout_step_times_sec)),
                "policy_row_mapping_sec": mapping_setup_sec,
            },
            "shape_assertions": "passed",
            "note": (
                "Uses a live strict native vector trainer env sample for "
                "observation and legal masks, then runs the existing synthetic "
                "Mctx search timing. This is loop-speed evidence only, not "
                "training quality, replay quality, or a broader CurvyTron mode."
            ),
        }
    else:
        obs_dim = _optional_positive_int(config, "obs_dim", hidden_dim)
        env_batch_size = None
        root_batch_size = batch_size
        candidate_ego_rows = root_batch_size
        live_ego_rows = root_batch_size
        host_observation_setup_sec = None
        host_to_device_transfer_sec = None
        obs = jnp.linspace(
            -1.0,
            1.0,
            root_batch_size * obs_dim,
            dtype=jnp.float32,
        ).reshape(root_batch_size, obs_dim)
        invalid_actions = jnp.zeros((root_batch_size, ACTION_COUNT), dtype=jnp.bool_)
        observation = {
            "mode": observation_mode,
            "source": "synthetic_flat_obs",
            "source_tensor_shape": [root_batch_size, obs_dim],
            "root_obs_shape": [root_batch_size, obs_dim],
            "note": (
                "Legacy fully synthetic root observation. Set "
                "--observation-mode curvytron_debug for the obs[B,P,9] shape."
            ),
        }

    transfer_warmup_times_sec: list[float] = []
    transfer_steady_times_sec: list[float] = []
    transfer_steady_median_sec = None
    if host_to_device_transfer_sec is not None:
        for _ in range(max(1, min(1, warmup_runs))):
            transfer_started = time.perf_counter()
            transfer_obs = jax.device_put(obs_host)
            transfer_invalid = jax.device_put(invalid_actions_host)
            transfer_obs.block_until_ready()
            transfer_invalid.block_until_ready()
            transfer_warmup_times_sec.append(time.perf_counter() - transfer_started)
        for _ in range(steady_runs):
            transfer_started = time.perf_counter()
            obs = jax.device_put(obs_host)
            invalid_actions = jax.device_put(invalid_actions_host)
            obs.block_until_ready()
            invalid_actions.block_until_ready()
            transfer_steady_times_sec.append(time.perf_counter() - transfer_started)
        transfer_steady_median_sec = statistics.median(transfer_steady_times_sec)

    params = {
        "representation_w": linspace_matrix(obs_dim, hidden_dim, 0.08),
        "representation_b": jnp.linspace(-0.02, 0.02, hidden_dim, dtype=jnp.float32),
        "action_embed": linspace_matrix(ACTION_COUNT, hidden_dim, 0.10),
        "dynamics_w": linspace_matrix(hidden_dim, hidden_dim, 0.025),
        "dynamics_b": jnp.linspace(0.01, -0.01, hidden_dim, dtype=jnp.float32),
        "policy_w": linspace_matrix(hidden_dim, ACTION_COUNT, 0.07),
        "policy_b": jnp.array([0.0, 0.03, -0.01], dtype=jnp.float32),
        "value_w": jnp.linspace(-0.04, 0.04, hidden_dim, dtype=jnp.float32),
        "reward_w": jnp.linspace(0.03, -0.03, hidden_dim, dtype=jnp.float32),
    }

    def prediction(params: dict[str, Any], hidden: Any) -> tuple[Any, Any]:
        prior_logits = hidden @ params["policy_w"] + params["policy_b"]
        value = jnp.tanh(hidden @ params["value_w"])
        return prior_logits, value

    def recurrent_fn(
        params: dict[str, Any],
        rng_key: Any,
        action: Any,
        hidden: Any,
    ) -> tuple[Any, Any]:
        del rng_key
        action_features = jax.nn.one_hot(action, ACTION_COUNT, dtype=jnp.float32)
        action_delta = action_features @ params["action_embed"]
        next_hidden = jnp.tanh(
            hidden + action_delta + hidden @ params["dynamics_w"] + params["dynamics_b"]
        )
        prior_logits, value = prediction(params, next_hidden)
        reward = 0.05 * jnp.tanh(next_hidden @ params["reward_w"])
        discount = jnp.full_like(value, 0.99)
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=value,
            ),
            next_hidden,
        )

    @functools.partial(jax.jit, static_argnames=("num_simulations", "max_depth"))
    def run_search(
        params: dict[str, Any],
        rng_key: Any,
        obs: Any,
        invalid_actions: Any,
        *,
        num_simulations: int,
        max_depth: int,
    ) -> Any:
        hidden = jnp.tanh(obs @ params["representation_w"] + params["representation_b"])
        prior_logits, value = prediction(params, hidden)
        root = mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=hidden)
        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=invalid_actions,
            max_depth=max_depth,
            max_num_considered_actions=ACTION_COUNT,
            gumbel_scale=1.0,
        )

    first_started = time.perf_counter()
    first_output = run_search(
        params,
        jax.random.PRNGKey(0),
        obs,
        invalid_actions,
        num_simulations=num_simulations,
        max_depth=max_depth,
    )
    first_output.action_weights.block_until_ready()
    compile_plus_first_run_sec = time.perf_counter() - first_started

    warmup_times_sec = []
    for run_index in range(warmup_runs):
        started = time.perf_counter()
        warmup_output = run_search(
            params,
            jax.random.PRNGKey(1 + run_index),
            obs,
            invalid_actions,
            num_simulations=num_simulations,
            max_depth=max_depth,
        )
        warmup_output.action_weights.block_until_ready()
        warmup_times_sec.append(time.perf_counter() - started)

    steady_times_sec = []
    output = first_output
    for run_index in range(steady_runs):
        started = time.perf_counter()
        output = run_search(
            params,
            jax.random.PRNGKey(1000 + run_index),
            obs,
            invalid_actions,
            num_simulations=num_simulations,
            max_depth=max_depth,
        )
        output.action_weights.block_until_ready()
        steady_times_sec.append(time.perf_counter() - started)

    steady_median_sec = statistics.median(steady_times_sec)
    steady_min_sec = min(steady_times_sec)
    steady_max_sec = max(steady_times_sec)
    decisions_per_sec_median = root_batch_size / steady_median_sec
    simulations_per_sec_median = (root_batch_size * num_simulations) / steady_median_sec
    env_rows_per_sec_median = (
        None if env_batch_size is None else env_batch_size / steady_median_sec
    )
    counts = {
        "env_rows": env_batch_size,
        "players_per_env": (
            player_count
            if observation_mode in CURVYTRON_SHAPED_OBSERVATION_MODES
            else None
        ),
        "candidate_ego_rows": candidate_ego_rows,
        "live_ego_rows": live_ego_rows,
        "search_root_count": root_batch_size,
        "action_count": ACTION_COUNT,
    }

    device_to_host_action_times_sec: list[float] = []
    device_to_host_action_weights_times_sec: list[float] = []
    for _ in range(max(1, steady_runs)):
        output_transfer_started = time.perf_counter()
        np.asarray(output.action)
        device_to_host_action_times_sec.append(
            time.perf_counter() - output_transfer_started
        )
        output_transfer_started = time.perf_counter()
        np.asarray(output.action_weights)
        device_to_host_action_weights_times_sec.append(
            time.perf_counter() - output_transfer_started
        )

    output_transfer_started = time.perf_counter()
    actions = np.asarray(output.action)
    action_weights = np.asarray(output.action_weights)
    device_to_host_output_transfer_sec = time.perf_counter() - output_transfer_started
    action_sample_count = min(root_batch_size, 128)
    row_sums = action_weights.sum(axis=1)
    finite_weights = bool(np.isfinite(action_weights).all())
    normalized_weights = bool(np.allclose(row_sums, 1.0, atol=1e-5))
    if not finite_weights:
        problems.append("action_weights contains non-finite values")
    if not normalized_weights:
        problems.append("action_weights rows do not sum to 1 within atol=1e-5")

    result = {
        "ok": not problems,
        "problems": problems,
        "packages": {
            "mctx": _version_or_missing("mctx"),
            "jax": _version_or_missing("jax"),
            "jaxlib": _version_or_missing("jaxlib"),
        },
        "jax": {
            "default_backend": backend,
            "devices": [str(device) for device in devices],
            "device_count": len(devices),
        },
        "nvidia_smi": _nvidia_smi(),
        "config": {
            "batch_size": batch_size,
            "env_batch_size": env_batch_size,
            "root_batch_size": root_batch_size,
            "player_count": player_count,
            "action_count": ACTION_COUNT,
            "observation_mode": observation_mode,
            "obs_dim": obs_dim,
            "hidden_dim": hidden_dim,
            "body_capacity": config.get("body_capacity"),
            "step_index": config.get("step_index"),
            "fixture_paths": _optional_fixture_paths(config),
            "group_id": config.get("group_id"),
            "event_mode": config.get("event_mode"),
            "rollout_steps": config.get("rollout_steps"),
            "seed": config.get("seed"),
            "decision_ms": config.get("decision_ms"),
            "actor_simulations": config.get("actor_simulations"),
            "actor_seed": config.get("actor_seed"),
            "allow_unverified": bool(config.get("allow_unverified", False)),
            "num_simulations": num_simulations,
            "max_depth": max_depth,
            "warmup_runs": warmup_runs,
            "steady_runs": steady_runs,
            "policy_kind": "gumbel_muzero_policy",
        },
        "counts": counts,
        "observation": observation,
        "timing": {
            "host_observation_setup_sec": host_observation_setup_sec,
            "host_to_device_transfer_sec": host_to_device_transfer_sec,
            "host_to_device_transfer_warmup_times_sec": transfer_warmup_times_sec,
            "host_to_device_transfer_steady_times_sec": transfer_steady_times_sec,
            "host_to_device_transfer_steady_median_sec": transfer_steady_median_sec,
            "compile_plus_first_run_sec": compile_plus_first_run_sec,
            "warmup_times_sec": warmup_times_sec,
            "steady_times_sec": steady_times_sec,
            "steady_median_sec": steady_median_sec,
            "steady_min_sec": steady_min_sec,
            "steady_max_sec": steady_max_sec,
            "device_to_host_output_transfer_sec": device_to_host_output_transfer_sec,
            "device_to_host_action_times_sec": device_to_host_action_times_sec,
            "device_to_host_action_median_sec": statistics.median(
                device_to_host_action_times_sec
            ),
            "device_to_host_action_weights_times_sec": (
                device_to_host_action_weights_times_sec
            ),
            "device_to_host_action_weights_median_sec": statistics.median(
                device_to_host_action_weights_times_sec
            ),
            "env_rows_per_sec_median": env_rows_per_sec_median,
            "decisions_per_sec_median": decisions_per_sec_median,
            "simulations_per_sec_median": simulations_per_sec_median,
        },
        "output": {
            "actions": actions[:action_sample_count].astype(int).tolist(),
            "actions_sample_count": action_sample_count,
            "actions_total_count": int(actions.shape[0]),
            "actions_truncated": bool(action_sample_count < actions.shape[0]),
            "action_histogram": np.bincount(actions, minlength=ACTION_COUNT)
            .astype(int)
            .tolist(),
            "action_weights_finite": finite_weights,
            "action_weights_normalized": normalized_weights,
            "action_weight_row_sum_min": float(row_sums.min()),
            "action_weight_row_sum_max": float(row_sums.max()),
            "action_weight_sample": action_weights[: min(root_batch_size, 4)]
            .astype(float)
            .tolist(),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.function(image=gpu_image, gpu=["L4", "T4"], timeout=10 * 60)
def mctx_synthetic_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    return _run_benchmark(config)


@app.local_entrypoint()
def main(
    batch_size: int = 16,
    num_simulations: int = 8,
    hidden_dim: int = 32,
    obs_dim: int = 0,
    observation_mode: str = OBSERVATION_MODE_FLAT,
    player_count: int = 2,
    body_capacity: int = 4,
    step_index: int = 0,
    fixture_paths: str = "",
    group_id: str = "",
    event_mode: str = "",
    rollout_steps: int = 2,
    seed: int = 123,
    decision_ms: float = 1000.0 / 60.0,
    actor_simulations: int = 4,
    actor_seed: int = 0,
    allow_unverified: bool = False,
    max_depth: int = 8,
    warmup_runs: int = 1,
    steady_runs: int = 5,
) -> None:
    config = {
        "batch_size": batch_size,
        "num_simulations": num_simulations,
        "hidden_dim": hidden_dim,
        "obs_dim": obs_dim,
        "observation_mode": observation_mode,
        "player_count": player_count,
        "body_capacity": body_capacity,
        "step_index": step_index,
        "fixture_paths": fixture_paths,
        "group_id": group_id,
        "rollout_steps": rollout_steps,
        "seed": seed,
        "decision_ms": decision_ms,
        "actor_simulations": actor_simulations,
        "actor_seed": actor_seed,
        "allow_unverified": allow_unverified,
        "max_depth": max_depth,
        "warmup_runs": warmup_runs,
        "steady_runs": steady_runs,
    }
    if event_mode:
        config["event_mode"] = event_mode
    result = mctx_synthetic_benchmark.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
