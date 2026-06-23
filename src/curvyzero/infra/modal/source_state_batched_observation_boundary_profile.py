"""Profile-only Modal sidecar for the batched observation boundary.

This does not modify trainers, tournaments, checkpoints, eval, Modal Volumes,
or live runs. Some profile-only probes may read an immutable checkpoint ref to
measure a real-model search boundary. It measures the candidate boundary around
the existing source-state GPU render prototype:

    CPU env step -> production-to-compact -> owner-ordered pack -> H2D
    -> fused two-view GPU render -> readback -> row-major conversion
    -> [B, 2, 4, 64, 64] stack update -> CPU parity checks

Run from repo root:

    uv run --extra modal modal run \
      -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import pickle
import statistics
import time
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import modal

from curvyzero.contracts.curvytron import curvytron_runs_volume_name
from curvyzero.contracts.curvytron import modal_volume_kwargs_for_name
from curvyzero.env import vector_runtime
from curvyzero.env.observation_surface_contract import POLICY_FRAME_STACK_DEPTH
from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.env.vector_multiplayer_env import SOURCE_PHYSICS_STEP_MS
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.infra.modal.mctx_dependency_smoke import MCTX_VERSION
from curvyzero.infra.modal.source_state_gpu_render_benchmark import APP_NAME as RENDER_APP_NAME
from curvyzero.infra.modal.source_state_gpu_render_benchmark import BONUS_RENDER_MODE_IDS
from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
    BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
)
from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
    COMPUTE_CHOICES as RENDER_COMPUTE_CHOICES,
)
from curvyzero.infra.modal.source_state_gpu_render_benchmark import COMPUTE_H100
from curvyzero.infra.modal.source_state_gpu_render_benchmark import COMPUTE_L4_T4
from curvyzero.infra.modal.source_state_gpu_render_benchmark import GEOMETRY_DTYPE_FLOAT32
from curvyzero.infra.modal.source_state_gpu_render_benchmark import GEOMETRY_DTYPE_FLOAT64
from curvyzero.infra.modal.source_state_gpu_render_benchmark import RENDER_MODE_BROWSER_LINES
from curvyzero.infra.modal.source_state_gpu_render_benchmark import RENDER_MODE_IDS
from curvyzero.infra.modal.source_state_gpu_render_benchmark import RENDER_SURFACE_BLOCK_704_GRAY64
from curvyzero.infra.modal.source_state_gpu_render_benchmark import RENDER_SURFACE_DIRECT_GRAY64
from curvyzero.infra.modal.source_state_gpu_render_benchmark import RENDER_VIEWS_BOTH
from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
    TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT,
)
from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
    TRAIL_COMPOSITION_PRIORITY_BUFFER,
)
from curvyzero.infra.modal.source_state_gpu_render_benchmark import _copy_state_to_device
from curvyzero.infra.modal.source_state_gpu_render_benchmark import _draw_direct_bonus_and_heads
from curvyzero.infra.modal.source_state_gpu_render_benchmark import _make_jax_two_view_render_fn
from curvyzero.infra.modal.source_state_gpu_render_benchmark import _nvidia_smi
from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
    _pack_compact_trails_in_owner_draw_order,
)
from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
    _production_to_benchmark_source_state,
)
from curvyzero.infra.modal.source_state_gpu_render_benchmark import _validate_config
from curvyzero.infra.modal.source_state_gpu_render_benchmark import gpu_image
from curvyzero.training.source_state_batched_observation_profile import (
    SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND,
)
from curvyzero.training.source_state_batched_observation_profile import (
    SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID,
)
from curvyzero.training.source_state_batched_observation_profile import (
    CpuOracleBatchedObservationRenderer,
)
from curvyzero.training.source_state_batched_observation_profile import (
    SourceStateBatchedRenderRequest,
)
from curvyzero.training.source_state_batched_observation_profile import (
    SourceStateBatchedRenderResult,
)
from curvyzero.training.source_state_batched_observation_profile import (
    source_state_render_state_with_row_overlays,
)
from curvyzero.training.source_state_batched_observation_mock_collector import (
    BatchedLightZeroProfileEnvManager,
    BatchedLightZeroScalarActionBridge,
)
from curvyzero.training.source_state_batched_observation_mock_collector import (
    materialize_lightzero_scalar_timestep,
)
from curvyzero.training.source_state_batched_observation_mock_collector import (
    materialize_trainer_surface_policy_timestep,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO,
    COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_TOY_PROBE,
    COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPLS,
    HybridBatchedStackProbeResult,
    HybridCompactBatch,
    HybridObservationProfileConfig,
    HybridPolicySearchProbeResult,
    HYBRID_STACK_STORAGE_DTYPES,
    PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_KEYS,
    PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_MARKER,
    run_hybrid_observation_profile,
)
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
)
from curvyzero.training.compact_policy_row_bridge import build_compact_root_batch_v1
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_replay_chunk_v1_from_search_result,
)
from curvyzero.training.compact_rollout_modes import COMPACT_ROLLOUT_SLAB_ACTION_MODES
from curvyzero.training.compact_rollout_modes import (
    COMPACT_ROLLOUT_SLAB_ACTION_MODE_SEARCH_FEEDBACK,
)
from curvyzero.training.compact_rollout_slab import CompactRolloutSlab
from curvyzero.training.compact_root_tape import InMemoryCompactRootTapeRecorderV1
from curvyzero.training.compact_root_tape import run_compact_root_tape_comparison_v1
from curvyzero.training.compact_search_service import CompactSearchComparatorServiceV1
from curvyzero.training.compact_search_service import CompactSearchServiceV1
from curvyzero.training.compact_search_service import compact_search_result_v1_from_arrays
from curvyzero.training.compact_torch_search_service import CompactTorchCompileConfig
from curvyzero.training.compact_torch_search_service import (
    COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
    COMPACT_TORCH_INITIAL_INFERENCE_MODE_MODEL_METHOD,
    COMPACT_TORCH_INITIAL_INFERENCE_MODES,
    COMPACT_TORCH_MEMORY_FORMATS,
    COMPACT_TORCH_MODEL_COMPILE_MODES,
)
from curvyzero.training.compact_torch_search_service import COMPACT_TORCH_TIMING_MODES
from curvyzero.training.compact_torch_search_service import CompactTorchSearchServiceV1
from curvyzero.training.compact_torch_search_service import COMPACT_TORCH_SEARCH_SERVICE_IMPL
from curvyzero.training.fixed_shape_batched_search_owner import (
    FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL,
)
from curvyzero.training.fixed_shape_batched_search_owner import (
    FixedShapeBatchedSearchOwnerV0,
)
from curvyzero.training.mctx_compact_search_service import MctxCompactSearchConfig
from curvyzero.training.mctx_compact_search_service import MctxCompactSearchServiceV1
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    TRAINER_STACK_BACKEND_CPU_DIRTY_CACHE,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
)
from curvyzero.training import exploration_bonus as xb


APP_NAME = "curvyzero-source-state-batched-observation-boundary-profile"
SCHEMA_ID = "curvyzero_source_state_batched_observation_boundary_profile/v0"
IMPL_ID = "curvyzero_modal_profile_only_batched_observation_boundary/v0"
RUNS_MOUNT = Path("/runs")
LIGHTZERO_VERSION = "0.2.0"
TORCH_CUDA12_VERSION = "2.8.0"
COMPUTE_H100_CPU64 = "gpu-h100-cpu64"
LIGHTZERO_CPU_HEAVY_COMPUTE_CHOICES = {COMPUTE_H100_CPU64}
COMPUTE_CHOICES = set(RENDER_COMPUTE_CHOICES) | LIGHTZERO_CPU_HEAVY_COMPUTE_CHOICES
TARGET_SIZE = 64
PLAYER_COUNT = 2
DEFAULT_BOUNDARY_GEOMETRY_DTYPE = GEOMETRY_DTYPE_FLOAT32
BOUNDARY_PARITY_MODE_AUTO = "auto"
BOUNDARY_PARITY_MODE_EXACT = "exact"
BOUNDARY_PARITY_MODE_TOLERANT = "tolerant"
BOUNDARY_PARITY_MODES = (
    BOUNDARY_PARITY_MODE_AUTO,
    BOUNDARY_PARITY_MODE_EXACT,
    BOUNDARY_PARITY_MODE_TOLERANT,
)
DEFAULT_BOUNDARY_PARITY_MODE = BOUNDARY_PARITY_MODE_AUTO
DEFAULT_BOUNDARY_PARITY_MAX_ABS_DIFF = 2
DEFAULT_BOUNDARY_PARITY_MAX_MISMATCH_FRACTION = 1.0e-4
DEFAULT_DYNAMIC_MIN_RENDER_TRAIL_SLOTS = 32
SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND = (
    "jax_gpu_persistent_policy_framebuffer_profile"
)
BOUNDARY_OBSERVATION_RENDERER_BACKENDS = (
    SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND,
    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND,
)
BOUNDARY_RENDER_SURFACES = (
    RENDER_SURFACE_BLOCK_704_GRAY64,
    RENDER_SURFACE_DIRECT_GRAY64,
)
LIGHTZERO_ARRAY_CEILING_MODE_POLICY_ARRAYS = "policy_arrays"
LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE = "mock_search_service"
LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE = "service_tax_probe"
LIGHTZERO_ARRAY_CEILING_MODE_RECURRENT_TOY = "recurrent_toy"
LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS = "dense_torch_mcts"
LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE = "dense_torch_mcts_compile_spike"
LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE = "compact_torch_search_service"
LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER = "fixed_shape_search_owner"
LIGHTZERO_ARRAY_CEILING_DENSE_TORCH_MCTS_MODES = (
    LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS,
    LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE,
)
LIGHTZERO_ARRAY_CEILING_MODES = (
    LIGHTZERO_ARRAY_CEILING_MODE_POLICY_ARRAYS,
    LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE,
    LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE,
    LIGHTZERO_ARRAY_CEILING_MODE_RECURRENT_TOY,
    LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS,
    LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE,
    LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE,
    LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER,
)
LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8 = "host_uint8"
LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8_PINNED = "host_uint8_pinned"
LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_FLOAT32 = "host_float32"
LIGHTZERO_ARRAY_CEILING_INPUT_MODE_RESIDENT_TORCH_REUSE = "resident_torch_reuse"
LIGHTZERO_ARRAY_CEILING_INPUT_MODES = (
    LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8,
    LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8_PINNED,
    LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_FLOAT32,
    LIGHTZERO_ARRAY_CEILING_INPUT_MODE_RESIDENT_TORCH_REUSE,
)
LIGHTZERO_MCTS_ARRAYS_BOUNDARY_SEMANTICS = "stock_lightzero_mcts_arrays_facade"
LIGHTZERO_MCTS_ARRAYS_BOUNDARY_DIRECT_SEMANTICS = "lightzero_mcts_arrays_direct_ctree_profile"
LIGHTZERO_MCTS_ARRAYS_BOUNDARY_GPU_LATENT_SEMANTICS = (
    "lightzero_mcts_arrays_direct_ctree_gpu_latent_profile"
)
LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_STOCK_FACADE = "stock_facade"
LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE = "direct_ctree_arrays"
LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT = "direct_ctree_gpu_latent"
LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT = (
    "direct_ctree_gpu_latent_precomputed_recurrent"
)
LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPLS = (
    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_STOCK_FACADE,
    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT,
)

TIMING_FIELDS = (
    "env_step_sec",
    "production_to_compact_sec",
    "owner_ordered_pack_sec",
    "host_to_device_sec",
    "device_render_sec",
    "device_to_host_sec",
    "view_major_to_row_major_sec",
    "stack_sec",
    "final_obs_sec",
    "autoreset_sec",
    "reset_render_sec",
    "reset_stack_sec",
    "candidate_total_observation_sec",
    "candidate_total_step_plus_observation_sec",
    "lightzero_scalarize_sec",
    "lightzero_payload_pickle_sec",
    "rnd_collect_data_sec",
    "rnd_train_with_data_sec",
    "rnd_estimate_sec",
    "candidate_total_mock_collector_sec",
    "cpu_reference_render_stack_sec",
    "lightzero_payload_pickle_bytes",
)
SURFACE_FACADE_DIVERGENCE_TIMING_FIELDS = (
    "candidate_surface_step_sec",
    "cpu_reference_render_stack_sec",
    "candidate_reset_sec",
    "cpu_reference_reset_render_stack_sec",
)
TRAIL_STAT_FIELDS = (
    "env_trail_slots",
    "max_render_trail_slots",
    "render_trail_slots",
    "active_trail_count_min",
    "active_trail_count_median",
    "active_trail_count_p95",
    "active_trail_count_max",
    "active_trail_count_sum",
    "active_trail_fraction_median",
    "active_trail_fraction_p95",
    "active_trail_fraction_max",
    "render_truncation_row_count",
    "render_truncation_row_fraction",
    "render_truncation_max_dropped_slots",
)
TRAIL_ARRAY_KEYS = (
    "trail_x",
    "trail_y",
    "trail_radius",
    "trail_owner",
    "trail_active",
    "trail_break_before",
)

app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name(
    curvytron_runs_volume_name(),
    **modal_volume_kwargs_for_name(curvytron_runs_volume_name()),
)
MIXED_JAX_TORCH_GPU_MEMORY_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
gpu_jax_torch_image = gpu_image.env(MIXED_JAX_TORCH_GPU_MEMORY_ENV)
gpu_rnd_image = gpu_jax_torch_image.uv_pip_install(f"torch=={TORCH_CUDA12_VERSION}")
gpu_lightzero_image = gpu_jax_torch_image.uv_pip_install(
    f"LightZero=={LIGHTZERO_VERSION}",
    f"torch=={TORCH_CUDA12_VERSION}",
    "cloudpickle>=3",
)
gpu_mctx_image = gpu_image.uv_pip_install(f"mctx=={MCTX_VERSION}")
gpu_mctx_lightzero_image = gpu_jax_torch_image.uv_pip_install(
    f"mctx=={MCTX_VERSION}",
    f"LightZero=={LIGHTZERO_VERSION}",
    f"torch=={TORCH_CUDA12_VERSION}",
    "cloudpickle>=3",
)


@app.function(image=gpu_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_boundary_profile(config: dict[str, Any]) -> dict[str, Any]:
    return _run_boundary_profile_safe(config)


@app.function(image=gpu_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_boundary_profile_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _run_boundary_profile_safe(config)


@app.function(image=gpu_rnd_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_boundary_profile_with_rnd(config: dict[str, Any]) -> dict[str, Any]:
    return _run_boundary_profile_safe(config)


@app.function(image=gpu_rnd_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_boundary_profile_h100_with_rnd(config: dict[str, Any]) -> dict[str, Any]:
    return _run_boundary_profile_safe(config)


@app.function(image=gpu_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_surface_facade_profile(config: dict[str, Any]) -> dict[str, Any]:
    return _run_surface_facade_profile_safe(config)


@app.function(image=gpu_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_surface_facade_profile_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _run_surface_facade_profile_safe(config)


@app.function(image=gpu_rnd_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_surface_facade_profile_with_rnd(config: dict[str, Any]) -> dict[str, Any]:
    return _run_surface_facade_profile_safe(config)


@app.function(image=gpu_rnd_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_surface_facade_profile_h100_with_rnd(config: dict[str, Any]) -> dict[str, Any]:
    return _run_surface_facade_profile_safe(config)


@app.function(image=gpu_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_profile_env_manager_profile(config: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_env_manager_profile_safe(config)


@app.function(image=gpu_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_profile_env_manager_profile_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_env_manager_profile_safe(config)


@app.function(image=gpu_rnd_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_profile_env_manager_profile_with_rnd(config: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_env_manager_profile_safe(config)


@app.function(image=gpu_rnd_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_profile_env_manager_profile_h100_with_rnd(config: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_env_manager_profile_safe(config)


@app.function(image=gpu_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_hybrid_observation_profile_modal(config: dict[str, Any]) -> dict[str, Any]:
    return _run_hybrid_observation_profile_safe(config)


@app.function(image=gpu_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_hybrid_observation_profile_modal_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _run_hybrid_observation_profile_safe(config)


@app.function(image=gpu_lightzero_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_hybrid_observation_profile_lightzero_modal(config: dict[str, Any]) -> dict[str, Any]:
    return _run_hybrid_observation_profile_safe(config)


@app.function(image=gpu_lightzero_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_hybrid_observation_profile_lightzero_modal_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _run_hybrid_observation_profile_safe(config)


@app.function(image=gpu_lightzero_image, gpu="H100", timeout=20 * 60, cpu=64.0)
def run_hybrid_observation_profile_lightzero_modal_h100_cpu64(
    config: dict[str, Any],
) -> dict[str, Any]:
    return _run_hybrid_observation_profile_safe(config)


@app.function(image=gpu_mctx_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_hybrid_observation_profile_mctx_modal(config: dict[str, Any]) -> dict[str, Any]:
    return _run_hybrid_observation_profile_safe(config)


@app.function(image=gpu_mctx_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_hybrid_observation_profile_mctx_modal_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _run_hybrid_observation_profile_safe(config)


@app.function(
    image=gpu_mctx_lightzero_image,
    gpu=["L4", "T4"],
    timeout=20 * 60,
    cpu=4.0,
    volumes={str(RUNS_MOUNT): runs_volume},
)
def run_hybrid_observation_profile_mctx_lightzero_modal(config: dict[str, Any]) -> dict[str, Any]:
    return _run_hybrid_observation_profile_safe(config)


@app.function(
    image=gpu_mctx_lightzero_image,
    gpu="H100",
    timeout=20 * 60,
    cpu=4.0,
    volumes={str(RUNS_MOUNT): runs_volume},
)
def run_hybrid_observation_profile_mctx_lightzero_modal_h100(
    config: dict[str, Any],
) -> dict[str, Any]:
    return _run_hybrid_observation_profile_safe(config)


def _run_boundary_profile_safe(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run_boundary_profile_impl(config)
    except Exception as exc:  # pragma: no cover - Modal remote diagnostics.
        import traceback

        return {
            "schema_id": SCHEMA_ID,
            "ok": False,
            "profile_only": True,
            "app_name": APP_NAME,
            "renderer_app_name": RENDER_APP_NAME,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "known_gaps": _known_gaps(),
        }


def _run_surface_facade_profile_safe(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run_surface_facade_profile_impl(config)
    except Exception as exc:  # pragma: no cover - Modal remote diagnostics.
        import traceback

        return {
            "schema_id": SCHEMA_ID,
            "ok": False,
            "profile_only": True,
            "surface_facade_canary": True,
            "app_name": APP_NAME,
            "renderer_app_name": RENDER_APP_NAME,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "known_gaps": _known_gaps(),
        }


def _run_profile_env_manager_profile_safe(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run_profile_env_manager_profile_impl(config)
    except Exception as exc:  # pragma: no cover - Modal remote diagnostics.
        import traceback

        return {
            "schema_id": SCHEMA_ID,
            "ok": False,
            "profile_only": True,
            "profile_env_manager_canary": True,
            "app_name": APP_NAME,
            "renderer_app_name": RENDER_APP_NAME,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "known_gaps": _known_gaps(),
        }


def _run_boundary_profile_impl(config: dict[str, Any]) -> dict[str, Any]:
    import jax
    import numpy as np

    if str(config.get("geometry_dtype", DEFAULT_BOUNDARY_GEOMETRY_DTYPE)) == "float64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    checked = _validate_boundary_config(np=np, config=config)
    render_mode_id = RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]
    bonus_render_mode_id = BONUS_RENDER_MODE_IDS[BONUS_RENDER_MODE_SIMPLE_SYMBOLS]
    render_fn_cache: dict[int, Any] = {}

    def render_fn_for_slots(render_trail_slots: int) -> Any:
        slots = int(render_trail_slots)
        cached = render_fn_cache.get(slots)
        if cached is not None:
            return cached
        slot_config = dict(checked["render_config"])
        slot_config["trail_slots"] = slots
        cached = _make_jax_two_view_render_fn(
            jax=jax,
            jnp=jnp,
            config=slot_config,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
        )
        render_fn_cache[slots] = cached
        return cached

    env = VectorMultiplayerEnv(
        checked["batch_size"],
        player_count=PLAYER_COUNT,
        seed=checked["seed"],
        decision_source_frames=1,
        source_physics_step_ms=SOURCE_PHYSICS_STEP_MS,
        body_capacity=checked["body_capacity"],
        map_size=checked["map_size"],
        natural_bonus_spawn=False,
        death_mode=checked["death_mode"],
        max_ticks=checked["max_ticks"],
    )
    cpu_renderer = CpuOracleBatchedObservationRenderer()
    rnd_model = _build_rnd_model(checked) if checked["include_rnd_meter"] else None
    reference_render_rows = _row_major_render_rows(np=np, batch_size=checked["batch_size"])
    reference_render_players = _row_major_render_players(np=np, batch_size=checked["batch_size"])
    reference_raw_out = np.zeros(
        (
            checked["batch_size"] * PLAYER_COUNT,
            1,
            TARGET_SIZE,
            TARGET_SIZE,
        ),
        dtype=np.uint8,
    )

    env.reset(seed=checked["seed"])
    candidate_stacks = np.zeros(
        (
            checked["batch_size"],
            PLAYER_COUNT,
            POLICY_FRAME_STACK_DEPTH,
            TARGET_SIZE,
            TARGET_SIZE,
        ),
        dtype=np.float32,
    )
    reference_stacks = np.zeros_like(candidate_stacks)

    reset_frames, reset_timings, reset_trail_stats = _render_candidate_frames(
        jax=jax,
        np=np,
        env=env,
        config=checked["render_config"],
        render_fn_for_slots=render_fn_for_slots,
    )
    reset_stack_sec = _push_row_major_frames_into_stack(candidate_stacks, reset_frames)
    reset_reference_frames, reset_reference_render_stack_sec = _render_cpu_reference_frames(
        renderer=cpu_renderer,
        env=env,
        row_indices=reference_render_rows,
        controlled_players=reference_render_players,
        out=reference_raw_out,
        reference_stacks=reference_stacks,
    )
    parity_summaries: list[dict[str, Any]] = []
    parity_summaries.append(
        _assert_parity(
            label="reset",
            candidate_frames=reset_frames,
            reference_frames=reset_reference_frames,
            candidate_stacks=candidate_stacks,
            reference_stacks=reference_stacks,
            config=checked,
        )
    )

    rng = np.random.default_rng(checked["seed"] + 91_337)
    per_step = {field: [] for field in TIMING_FIELDS}
    per_step_trail_stats = {field: [] for field in TRAIL_STAT_FIELDS}
    parity_checks = 1
    first_loop_render_sec: float | None = None
    steady_start = checked["warmup_steps"]
    steady_terminal_step_count = 0
    steady_terminal_row_count = 0

    for iteration in range(checked["warmup_steps"] + checked["steps"]):
        controlled_actions = rng.integers(
            0,
            ACTION_COUNT,
            size=(checked["batch_size"],),
            dtype=np.int16,
        )
        other_actions = rng.integers(
            0,
            ACTION_COUNT,
            size=(checked["batch_size"],),
            dtype=np.int16,
        )
        joint_actions = _joint_actions_for_profile(
            np=np,
            controlled_actions=controlled_actions,
            other_actions=other_actions,
            batch_size=checked["batch_size"],
        )

        started = time.perf_counter()
        step_batch = env.step(joint_actions)
        env_step_sec = time.perf_counter() - started

        frames, timings, trail_stats = _render_candidate_frames(
            jax=jax,
            np=np,
            env=env,
            config=checked["render_config"],
            render_fn_for_slots=render_fn_for_slots,
        )
        stack_sec = _push_row_major_frames_into_stack(candidate_stacks, frames)
        needs_parity_reference = (
            iteration >= steady_start and parity_checks <= checked["verify_steps"]
        )
        measured_iteration = iteration - steady_start
        needs_timing_reference = (
            iteration >= steady_start
            and checked["cpu_reference_interval"] > 0
            and measured_iteration % checked["cpu_reference_interval"] == 0
        )
        reference_frames = None
        cpu_reference_render_stack_sec = 0.0
        if needs_parity_reference or needs_timing_reference:
            reference_frames, cpu_reference_render_stack_sec = _render_cpu_reference_frames(
                renderer=cpu_renderer,
                env=env,
                row_indices=reference_render_rows,
                controlled_players=reference_render_players,
                out=reference_raw_out,
                reference_stacks=reference_stacks,
            )

        if first_loop_render_sec is None:
            first_loop_render_sec = timings["device_render_sec"]

        if needs_parity_reference:
            if reference_frames is None:
                raise RuntimeError("internal error: parity reference was not rendered")
            parity_summaries.append(
                _assert_parity(
                    label=f"step_{iteration - steady_start}",
                    candidate_frames=frames,
                    reference_frames=reference_frames,
                    candidate_stacks=candidate_stacks,
                    reference_stacks=reference_stacks,
                    config=checked,
                )
            )
            parity_checks += 1

        final_obs_sec = 0.0
        autoreset_sec = 0.0
        reset_render_sec = 0.0
        reset_stack_sec = 0.0
        lightzero_scalarize_sec = 0.0
        lightzero_payload_pickle_sec = 0.0
        lightzero_payload_pickle_bytes = 0
        rnd_collect_data_sec = 0.0
        rnd_train_with_data_sec = 0.0
        rnd_estimate_sec = 0.0
        done_mask = env.state["done"].copy()
        candidate_final_observation = None
        if bool(done_mask.any()):
            if iteration >= steady_start:
                steady_terminal_step_count += 1
                steady_terminal_row_count += int(done_mask.sum())
            started = time.perf_counter()
            candidate_final_observation = np.zeros_like(candidate_stacks)
            candidate_final_observation[done_mask] = candidate_stacks[done_mask]
            if checked["cpu_reference_interval"] == 1:
                reference_final_observation = np.zeros_like(reference_stacks)
                reference_final_observation[done_mask] = reference_stacks[done_mask]
                parity_summaries.append(
                    _assert_final_observation_parity(
                        label=f"terminal_{iteration - steady_start}",
                        candidate_final_observation=candidate_final_observation,
                        reference_final_observation=reference_final_observation,
                        done_mask=done_mask,
                        config=checked,
                    )
                )
            final_obs_sec = time.perf_counter() - started

        if checked["include_lightzero_payload_profile"]:
            started = time.perf_counter()
            timestep, flat_obs, target_reward = materialize_lightzero_scalar_timestep(
                step_observation=candidate_stacks,
                step_reward=step_batch.reward,
                step_done=done_mask,
                final_observation=candidate_final_observation,
                batch_size=checked["batch_size"],
                player_count=PLAYER_COUNT,
            )
            lightzero_scalarize_sec = time.perf_counter() - started
            if checked["pickle_lightzero_payload"]:
                started = time.perf_counter()
                payload = pickle.dumps(timestep, protocol=pickle.HIGHEST_PROTOCOL)
                lightzero_payload_pickle_sec = time.perf_counter() - started
                lightzero_payload_pickle_bytes = len(payload)
            if rnd_model is not None:
                segment = SimpleNamespace(obs_segment=flat_obs)
                started = time.perf_counter()
                rnd_model.collect_data([[segment]])
                rnd_collect_data_sec = time.perf_counter() - started

                started = time.perf_counter()
                rnd_model.train_with_data()
                rnd_train_with_data_sec = time.perf_counter() - started

                started = time.perf_counter()
                rnd_model.estimate([[flat_obs], [target_reward]])
                rnd_estimate_sec = time.perf_counter() - started

        if bool(done_mask.any()):
            started = time.perf_counter()
            env.autoreset_done_rows(
                seed=checked["seed"] + 1_000_003 + iteration,
                row_mask=done_mask,
            )
            autoreset_sec = time.perf_counter() - started

            (
                reset_frames_after_terminal,
                reset_timings_after_terminal,
                _reset_trail_stats_after_terminal,
            ) = _render_candidate_frames(
                jax=jax,
                np=np,
                env=env,
                config=checked["render_config"],
                render_fn_for_slots=render_fn_for_slots,
            )
            reset_render_sec = _candidate_render_total_sec(reset_timings_after_terminal)
            reset_stack_sec = _push_row_major_frames_into_stack(
                candidate_stacks,
                reset_frames_after_terminal,
                row_mask=done_mask,
                reset_selected_rows=True,
            )
            if checked["cpu_reference_interval"] == 1:
                (
                    reset_reference_frames_after_terminal,
                    reset_reference_render_stack_sec,
                ) = _render_cpu_reference_frames(
                    renderer=cpu_renderer,
                    env=env,
                    row_indices=reference_render_rows,
                    controlled_players=reference_render_players,
                    out=reference_raw_out,
                    reference_stacks=reference_stacks,
                    row_mask=done_mask,
                    reset_selected_rows=True,
                )
                parity_summaries.append(
                    _assert_parity(
                        label=f"autoreset_{iteration - steady_start}",
                        candidate_frames=reset_frames_after_terminal,
                        reference_frames=reset_reference_frames_after_terminal,
                        candidate_stacks=candidate_stacks,
                        reference_stacks=reference_stacks,
                        config=checked,
                    )
                )
                cpu_reference_render_stack_sec += reset_reference_render_stack_sec

        if iteration >= steady_start:
            observation_sec = (
                timings["production_to_compact_sec"]
                + timings["owner_ordered_pack_sec"]
                + timings["host_to_device_sec"]
                + timings["device_render_sec"]
                + timings["device_to_host_sec"]
                + timings["view_major_to_row_major_sec"]
                + stack_sec
                + final_obs_sec
                + autoreset_sec
                + reset_render_sec
                + reset_stack_sec
            )
            per_step["env_step_sec"].append(env_step_sec)
            per_step["production_to_compact_sec"].append(timings["production_to_compact_sec"])
            per_step["owner_ordered_pack_sec"].append(timings["owner_ordered_pack_sec"])
            per_step["host_to_device_sec"].append(timings["host_to_device_sec"])
            per_step["device_render_sec"].append(timings["device_render_sec"])
            per_step["device_to_host_sec"].append(timings["device_to_host_sec"])
            per_step["view_major_to_row_major_sec"].append(timings["view_major_to_row_major_sec"])
            per_step["stack_sec"].append(stack_sec)
            per_step["final_obs_sec"].append(final_obs_sec)
            per_step["autoreset_sec"].append(autoreset_sec)
            per_step["reset_render_sec"].append(reset_render_sec)
            per_step["reset_stack_sec"].append(reset_stack_sec)
            per_step["candidate_total_observation_sec"].append(observation_sec)
            per_step["candidate_total_step_plus_observation_sec"].append(
                env_step_sec + observation_sec
            )
            payload_sec = (
                lightzero_scalarize_sec
                + lightzero_payload_pickle_sec
                + rnd_collect_data_sec
                + rnd_train_with_data_sec
                + rnd_estimate_sec
            )
            per_step["lightzero_scalarize_sec"].append(lightzero_scalarize_sec)
            per_step["lightzero_payload_pickle_sec"].append(lightzero_payload_pickle_sec)
            per_step["rnd_collect_data_sec"].append(rnd_collect_data_sec)
            per_step["rnd_train_with_data_sec"].append(rnd_train_with_data_sec)
            per_step["rnd_estimate_sec"].append(rnd_estimate_sec)
            per_step["candidate_total_mock_collector_sec"].append(
                env_step_sec + observation_sec + payload_sec
            )
            per_step["cpu_reference_render_stack_sec"].append(cpu_reference_render_stack_sec)
            per_step["lightzero_payload_pickle_bytes"].append(float(lightzero_payload_pickle_bytes))
            for field in TRAIL_STAT_FIELDS:
                per_step_trail_stats[field].append(float(trail_stats[field]))

    return {
        "schema_id": SCHEMA_ID,
        "impl_id": IMPL_ID,
        "ok": True,
        "profile_only": True,
        "app_name": APP_NAME,
        "renderer_app_name": RENDER_APP_NAME,
        "nvidia_smi": _nvidia_smi(),
        "packages": {
            "jax": _version_or_missing("jax"),
            "jaxlib": _version_or_missing("jaxlib"),
            "numpy": _version_or_missing("numpy"),
            "torch": _version_or_missing("torch"),
        },
        "jax": {
            "default_backend": jax.default_backend(),
            "device_count": len(jax.devices()),
            "devices": [str(device) for device in jax.devices()],
        },
        "config": checked,
        "timings": _summarize_timings(per_step),
        "trail_stats": _summarize_numeric(per_step_trail_stats),
        "warmup": {
            "warmup_steps": checked["warmup_steps"],
            "reset_render_sec_includes_first_jit_compile": reset_timings["device_render_sec"],
            "first_loop_render_sec": first_loop_render_sec,
            "reset_boundary_timings": {
                **reset_timings,
                "stack_sec": reset_stack_sec,
                "cpu_reference_render_stack_sec": reset_reference_render_stack_sec,
            },
            "reset_trail_stats": reset_trail_stats,
        },
        "parity": {
            "ok": True,
            **_summarize_parity(parity_summaries, config=checked),
            "checked_reset": True,
            "checked_step_count": max(0, parity_checks - 1),
            "frame_shape": [checked["batch_size"], PLAYER_COUNT, 1, TARGET_SIZE, TARGET_SIZE],
            "stack_shape": [
                checked["batch_size"],
                PLAYER_COUNT,
                POLICY_FRAME_STACK_DEPTH,
                TARGET_SIZE,
                TARGET_SIZE,
            ],
        },
        "terminal": {
            "steady_terminal_step_count": steady_terminal_step_count,
            "steady_terminal_row_count": steady_terminal_row_count,
            "terminal_profile": checked["max_ticks"] < 2_000,
            "max_ticks": checked["max_ticks"],
        },
        "mock_collector": {
            "included": checked["include_lightzero_payload_profile"],
            "pickle_lightzero_payload": checked["pickle_lightzero_payload"],
            "include_rnd_meter": checked["include_rnd_meter"],
            "rnd_metrics": None
            if rnd_model is None
            else rnd_model.metrics_snapshot(reason="boundary_profile"),
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
        },
        "known_gaps": _known_gaps(),
    }


def _run_hybrid_observation_profile_safe(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run_hybrid_observation_profile_impl(config)
    except Exception as exc:  # pragma: no cover - Modal remote diagnostics.
        import traceback

        return {
            "schema_id": SCHEMA_ID,
            "ok": False,
            "profile_only": True,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "hybrid_observation_canary": True,
            "app_name": APP_NAME,
            "renderer_app_name": RENDER_APP_NAME,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "known_gaps": _known_gaps(),
        }


def _select_profile_function(
    *,
    compute: str,
    hybrid_observation_canary: bool,
    hybrid_mctx_compact_search_probe: bool,
    hybrid_compact_root_tape_compare_mctx: bool,
    hybrid_compact_root_tape_compare_model_compile: bool,
    hybrid_compact_root_tape_compare_direct_core: bool,
    hybrid_mctx_lightzero_checkpoint_ref: str,
    hybrid_lightzero_collect_forward_probe: bool,
    hybrid_lightzero_initial_inference_probe: bool,
    hybrid_lightzero_array_ceiling_probe: bool,
    hybrid_lightzero_mcts_arrays_boundary_probe: bool,
    profile_env_manager_canary: bool,
    surface_facade_canary: bool,
    include_rnd_meter: bool,
) -> Any:
    lightzero_probe = (
        hybrid_lightzero_collect_forward_probe
        or hybrid_lightzero_initial_inference_probe
        or hybrid_lightzero_array_ceiling_probe
        or hybrid_lightzero_mcts_arrays_boundary_probe
    )
    if hybrid_observation_canary:
        if hybrid_mctx_compact_search_probe:
            if hybrid_mctx_lightzero_checkpoint_ref:
                return (
                    run_hybrid_observation_profile_mctx_lightzero_modal_h100
                    if compute == COMPUTE_H100
                    else run_hybrid_observation_profile_mctx_lightzero_modal
                )
            return (
                run_hybrid_observation_profile_mctx_modal_h100
                if compute == COMPUTE_H100
                else run_hybrid_observation_profile_mctx_modal
            )
        if hybrid_compact_root_tape_compare_mctx:
            if lightzero_probe:
                return (
                    run_hybrid_observation_profile_mctx_lightzero_modal_h100
                    if compute == COMPUTE_H100
                    else run_hybrid_observation_profile_mctx_lightzero_modal
                )
            return (
                run_hybrid_observation_profile_mctx_modal_h100
                if compute == COMPUTE_H100
                else run_hybrid_observation_profile_mctx_modal
            )
        if (
            hybrid_compact_root_tape_compare_model_compile
            or hybrid_compact_root_tape_compare_direct_core
        ) and lightzero_probe:
            if compute == COMPUTE_H100_CPU64:
                return run_hybrid_observation_profile_lightzero_modal_h100_cpu64
            if compute == COMPUTE_H100:
                return run_hybrid_observation_profile_lightzero_modal_h100
            return run_hybrid_observation_profile_lightzero_modal
        if lightzero_probe:
            if compute == COMPUTE_H100_CPU64:
                return run_hybrid_observation_profile_lightzero_modal_h100_cpu64
            if compute == COMPUTE_H100:
                return run_hybrid_observation_profile_lightzero_modal_h100
            return run_hybrid_observation_profile_lightzero_modal
        return (
            run_hybrid_observation_profile_modal_h100
            if compute == COMPUTE_H100
            else run_hybrid_observation_profile_modal
        )
    if profile_env_manager_canary and include_rnd_meter:
        return (
            run_profile_env_manager_profile_h100_with_rnd
            if compute == COMPUTE_H100
            else run_profile_env_manager_profile_with_rnd
        )
    if profile_env_manager_canary:
        return (
            run_profile_env_manager_profile_h100
            if compute == COMPUTE_H100
            else run_profile_env_manager_profile
        )
    if surface_facade_canary and include_rnd_meter:
        return (
            run_surface_facade_profile_h100_with_rnd
            if compute == COMPUTE_H100
            else run_surface_facade_profile_with_rnd
        )
    if surface_facade_canary:
        return (
            run_surface_facade_profile_h100
            if compute == COMPUTE_H100
            else run_surface_facade_profile
        )
    if include_rnd_meter:
        return (
            run_boundary_profile_h100_with_rnd
            if compute == COMPUTE_H100
            else run_boundary_profile_with_rnd
        )
    return run_boundary_profile_h100 if compute == COMPUTE_H100 else run_boundary_profile


def _runs_checkpoint_path(checkpoint_ref: str) -> Path:
    raw = Path(checkpoint_ref)
    if raw.is_absolute():
        resolved = raw.resolve()
        try:
            resolved.relative_to(RUNS_MOUNT)
        except ValueError as exc:
            raise ValueError(
                f"absolute checkpoint ref must live under {RUNS_MOUNT}: {checkpoint_ref!r}"
            ) from exc
        return resolved
    return (RUNS_MOUNT / raw).resolve()


def _build_mctx_lightzero_shadow_model(
    *,
    checkpoint_ref: str,
    state_key: str | None,
    seed: int,
    num_simulations: int,
    batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    from curvyzero.training.lightzero_checkpoint_opponent_provider import (
        load_lightzero_curvytron_visual_survival_policy,
    )
    from curvyzero.training.lightzero_jax_shadow_model_parity import (
        checkpoint_sha256,
        jax_shadow_from_state_dict,
        require_immutable_checkpoint_ref,
    )

    immutable_ref = require_immutable_checkpoint_ref(checkpoint_ref)
    checkpoint_path = _runs_checkpoint_path(immutable_ref)
    policy, device, load_summary = load_lightzero_curvytron_visual_survival_policy(
        checkpoint_path=checkpoint_path,
        seed=seed,
        num_simulations=num_simulations,
        batch_size=batch_size,
        use_cuda=False,
        state_key=state_key,
    )
    model = getattr(policy, "_model")
    if hasattr(model, "eval"):
        model.eval()
    shadow = jax_shadow_from_state_dict(model.state_dict())
    metadata = {
        "checkpoint_ref": immutable_ref,
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_state_key": state_key,
        "checkpoint_load_summary": load_summary,
        "torch_load_device": str(device),
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "action_space_size": int(shadow.action_space_size),
        "reward_support_size": int(shadow.reward_support_size),
        "value_support_size": int(shadow.value_support_size),
        "coverage_note": (
            "Full shadow coverage is only meaningful after the first search call; "
            "see mctx_compact_search_service_shadow_coverage."
        ),
    }
    return shadow, metadata


def _build_checkpoint_direct_ctree_compact_service(
    *,
    checkpoint_ref: str,
    state_key: str | None,
    seed: int,
    num_simulations: int,
    batch_size: int,
    use_cuda: bool,
    arrays_boundary_impl: str,
    input_mode: str,
    temperature: float,
    epsilon: float,
) -> _LightZeroCollectForwardCompactSearchService:
    from curvyzero.training.lightzero_checkpoint_opponent_provider import (
        load_lightzero_curvytron_visual_survival_policy,
    )
    from curvyzero.training.lightzero_jax_shadow_model_parity import (
        checkpoint_sha256,
        require_immutable_checkpoint_ref,
    )

    immutable_ref = require_immutable_checkpoint_ref(checkpoint_ref)
    checkpoint_path = _runs_checkpoint_path(immutable_ref)
    policy, device, load_summary = load_lightzero_curvytron_visual_survival_policy(
        checkpoint_path=checkpoint_path,
        seed=seed,
        num_simulations=num_simulations,
        batch_size=batch_size,
        use_cuda=use_cuda,
        state_key=state_key,
    )
    cfg = getattr(policy, "_cfg", None)
    if cfg is not None:
        cfg.root_noise_weight = 0.0
        eps_cfg = getattr(cfg, "eps", None)
        if eps_cfg is not None:
            eps_cfg.eps_greedy_exploration_in_collect = True
    model = getattr(policy, "_model", None)
    if model is not None and hasattr(model, "eval"):
        model.eval()
    policy_metadata = {
        "policy": policy,
        "build_sec": 0.0,
        "requested_cuda": bool(use_cuda),
        "cuda": str(device).startswith("cuda"),
        "policy_class": f"{type(policy).__module__}.{type(policy).__name__}",
        "model_class": f"{type(model).__module__}.{type(model).__name__}"
        if model is not None
        else None,
        "policy_device": str(device),
        "num_simulations": int(num_simulations),
        "collect_with_pure_policy": False,
        "root_noise_weight": 0.0,
        "batch_size": int(batch_size),
        "checkpoint_ref": immutable_ref,
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_state_key": state_key,
        "checkpoint_load_summary": load_summary,
        "surface": {
            "env_variant": "source_state_fixed_opponent",
            "observation_shape": [4, 64, 64],
        },
    }
    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata=policy_metadata,
        num_simulations=num_simulations,
        temperature=temperature,
        epsilon=epsilon,
        arrays_boundary=True,
        arrays_boundary_impl=arrays_boundary_impl,
        input_mode=input_mode,
    )
    return _LightZeroCollectForwardCompactSearchService(probe)


def _validate_boundary_config(*, np: Any, config: dict[str, Any]) -> dict[str, Any]:
    batch_size = _positive_int(config.get("batch_size", 64), "batch_size")
    actor_count = _positive_int(
        config.get("actor_count", min(4, batch_size)), "actor_count"
    )
    if actor_count > batch_size:
        raise ValueError("actor_count cannot exceed batch_size")
    compute = str(config.get("compute", COMPUTE_H100))
    if compute not in COMPUTE_CHOICES:
        allowed = ", ".join(sorted(COMPUTE_CHOICES))
        raise ValueError(f"compute must be one of {allowed}; got {compute!r}")
    seed = int(config.get("seed", 20260515))
    trail_slots = _positive_int(config.get("trail_slots", 256), "trail_slots")
    body_capacity = _positive_int(config.get("body_capacity", trail_slots), "body_capacity")
    if body_capacity < trail_slots:
        raise ValueError(
            "body_capacity must be greater than or equal to trail_slots; "
            f"got body_capacity={body_capacity}, trail_slots={trail_slots}"
        )
    steps = _positive_int(config.get("steps", 8), "steps")
    warmup_steps = _nonnegative_int(config.get("warmup_steps", 2), "warmup_steps")
    verify_steps = _nonnegative_int(config.get("verify_steps", 2), "verify_steps")
    cpu_reference_interval = _nonnegative_int(
        config.get("cpu_reference_interval", 1),
        "cpu_reference_interval",
    )
    if cpu_reference_interval != 1 and verify_steps > 0:
        raise ValueError(
            "cpu_reference_interval must be 1 when verify_steps > 0 so reference "
            "stack history stays aligned; use verify_steps=0 for candidate-only timing rows"
        )
    max_ticks_config = config.get("max_ticks", None)
    max_ticks = 2_000 if max_ticks_config is None else _positive_int(max_ticks_config, "max_ticks")
    death_mode = str(config.get("death_mode", vector_runtime.DEATH_MODE_PROFILE_NO_DEATH))
    if death_mode not in vector_runtime.DEATH_MODES:
        allowed = ", ".join(sorted(vector_runtime.DEATH_MODES))
        raise ValueError(f"death_mode must be one of {allowed}; got {death_mode!r}")
    include_lightzero_payload_profile = _bool_config(
        config.get("include_lightzero_payload_profile", False),
        "include_lightzero_payload_profile",
    )
    pickle_lightzero_payload = _bool_config(
        config.get("pickle_lightzero_payload", True),
        "pickle_lightzero_payload",
    )
    include_rnd_meter = _bool_config(
        config.get("include_rnd_meter", False),
        "include_rnd_meter",
    )
    surface_facade_canary = _bool_config(
        config.get("surface_facade_canary", False),
        "surface_facade_canary",
    )
    profile_env_manager_canary = _bool_config(
        config.get("profile_env_manager_canary", False),
        "profile_env_manager_canary",
    )
    surface_stack_backend = str(
        config.get("surface_stack_backend", TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE)
    )
    if surface_stack_backend not in {
        TRAINER_STACK_BACKEND_CPU_DIRTY_CACHE,
        TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
    }:
        raise ValueError(
            "surface_stack_backend must be "
            f"{TRAINER_STACK_BACKEND_CPU_DIRTY_CACHE!r} or "
            f"{TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE!r}; "
            f"got {surface_stack_backend!r}"
        )
    observation_renderer_backend = str(
        config.get(
            "observation_renderer_backend",
            SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND,
        )
    )
    if observation_renderer_backend not in BOUNDARY_OBSERVATION_RENDERER_BACKENDS:
        allowed = ", ".join(BOUNDARY_OBSERVATION_RENDERER_BACKENDS)
        raise ValueError(
            "observation_renderer_backend must be one of "
            f"{allowed}; got {observation_renderer_backend!r}"
        )
    async_device_only_profile = _bool_config(
        config.get("async_device_only_profile", False),
        "async_device_only_profile",
    )
    render_surface = str(config.get("render_surface", RENDER_SURFACE_BLOCK_704_GRAY64))
    if render_surface not in BOUNDARY_RENDER_SURFACES:
        allowed = ", ".join(BOUNDARY_RENDER_SURFACES)
        raise ValueError(f"render_surface must be one of {allowed}; got {render_surface!r}")
    hybrid_observation_canary = _bool_config(
        config.get("hybrid_observation_canary", False),
        "hybrid_observation_canary",
    )
    hybrid_lightzero_collect_forward_probe = _bool_config(
        config.get("hybrid_lightzero_collect_forward_probe", False),
        "hybrid_lightzero_collect_forward_probe",
    )
    hybrid_lightzero_initial_inference_probe = _bool_config(
        config.get("hybrid_lightzero_initial_inference_probe", False),
        "hybrid_lightzero_initial_inference_probe",
    )
    hybrid_lightzero_array_ceiling_probe = _bool_config(
        config.get("hybrid_lightzero_array_ceiling_probe", False),
        "hybrid_lightzero_array_ceiling_probe",
    )
    hybrid_lightzero_array_ceiling_mode = str(
        config.get(
            "hybrid_lightzero_array_ceiling_mode",
            LIGHTZERO_ARRAY_CEILING_MODE_POLICY_ARRAYS,
        )
    )
    if hybrid_lightzero_array_ceiling_mode not in LIGHTZERO_ARRAY_CEILING_MODES:
        allowed = ", ".join(LIGHTZERO_ARRAY_CEILING_MODES)
        raise ValueError(
            "hybrid_lightzero_array_ceiling_mode must be one of "
            f"{allowed}; got {hybrid_lightzero_array_ceiling_mode!r}"
        )
    hybrid_lightzero_mcts_arrays_boundary_probe = _bool_config(
        config.get("hybrid_lightzero_mcts_arrays_boundary_probe", False),
        "hybrid_lightzero_mcts_arrays_boundary_probe",
    )
    hybrid_mctx_compact_search_probe = _bool_config(
        config.get("hybrid_mctx_compact_search_probe", False),
        "hybrid_mctx_compact_search_probe",
    )
    hybrid_mctx_lightzero_checkpoint_ref = str(
        config.get("hybrid_mctx_lightzero_checkpoint_ref", "")
    ).strip()
    hybrid_mctx_lightzero_checkpoint_state_key = str(
        config.get("hybrid_mctx_lightzero_checkpoint_state_key", "")
    ).strip()
    hybrid_mctx_compare_direct_ctree = _bool_config(
        config.get("hybrid_mctx_compare_direct_ctree", False),
        "hybrid_mctx_compare_direct_ctree",
    )
    hybrid_mctx_compare_direct_ctree_impl = str(
        config.get(
            "hybrid_mctx_compare_direct_ctree_impl",
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
        )
    )
    if hybrid_mctx_compare_direct_ctree_impl not in {
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT,
    }:
        allowed = ", ".join(
            [
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT,
            ]
        )
        raise ValueError(
            "hybrid_mctx_compare_direct_ctree_impl must be one of "
            f"{allowed}; got {hybrid_mctx_compare_direct_ctree_impl!r}"
        )
    if hybrid_mctx_lightzero_checkpoint_ref and not hybrid_mctx_compact_search_probe:
        raise ValueError(
            "hybrid_mctx_lightzero_checkpoint_ref requires hybrid_mctx_compact_search_probe"
        )
    if hybrid_mctx_lightzero_checkpoint_state_key and not hybrid_mctx_lightzero_checkpoint_ref:
        raise ValueError(
            "hybrid_mctx_lightzero_checkpoint_state_key requires "
            "hybrid_mctx_lightzero_checkpoint_ref"
        )
    if hybrid_mctx_compare_direct_ctree and not hybrid_mctx_lightzero_checkpoint_ref:
        raise ValueError(
            "hybrid_mctx_compare_direct_ctree requires "
            "hybrid_mctx_lightzero_checkpoint_ref so both services use the same checkpoint"
        )
    hybrid_lightzero_mcts_arrays_boundary_impl = str(
        config.get(
            "hybrid_lightzero_mcts_arrays_boundary_impl",
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_STOCK_FACADE,
        )
    )
    if hybrid_lightzero_mcts_arrays_boundary_impl not in LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPLS:
        allowed = ", ".join(LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPLS)
        raise ValueError(
            "hybrid_lightzero_mcts_arrays_boundary_impl must be one of "
            f"{allowed}; got {hybrid_lightzero_mcts_arrays_boundary_impl!r}"
        )
    hybrid_lightzero_mcts_arrays_boundary_input_mode = str(
        config.get(
            "hybrid_lightzero_mcts_arrays_boundary_input_mode",
            LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8,
        )
    )
    if hybrid_lightzero_mcts_arrays_boundary_input_mode not in LIGHTZERO_ARRAY_CEILING_INPUT_MODES:
        allowed = ", ".join(LIGHTZERO_ARRAY_CEILING_INPUT_MODES)
        raise ValueError(
            "hybrid_lightzero_mcts_arrays_boundary_input_mode must be one of "
            f"{allowed}; got {hybrid_lightzero_mcts_arrays_boundary_input_mode!r}"
        )
    hybrid_lightzero_array_ceiling_input_mode = str(
        config.get(
            "hybrid_lightzero_array_ceiling_input_mode",
            LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8,
        )
    )
    if hybrid_lightzero_array_ceiling_input_mode not in LIGHTZERO_ARRAY_CEILING_INPUT_MODES:
        allowed = ", ".join(LIGHTZERO_ARRAY_CEILING_INPUT_MODES)
        raise ValueError(
            "hybrid_lightzero_array_ceiling_input_mode must be one of "
            f"{allowed}; got {hybrid_lightzero_array_ceiling_input_mode!r}"
        )
    hybrid_compact_rollout_slab_action_mode = str(
        config.get(
            "hybrid_compact_rollout_slab_action_mode",
            COMPACT_ROLLOUT_SLAB_ACTION_MODE_SEARCH_FEEDBACK,
        )
    )
    if hybrid_compact_rollout_slab_action_mode not in COMPACT_ROLLOUT_SLAB_ACTION_MODES:
        allowed = ", ".join(COMPACT_ROLLOUT_SLAB_ACTION_MODES)
        raise ValueError(
            "hybrid_compact_rollout_slab_action_mode must be one of "
            f"{allowed}; got {hybrid_compact_rollout_slab_action_mode!r}"
        )
    hybrid_compact_rollout_slab_probe = _bool_config(
        config.get("hybrid_compact_rollout_slab_probe", False),
        "hybrid_compact_rollout_slab_probe",
    )
    hybrid_compact_root_tape_compare = _bool_config(
        config.get("hybrid_compact_root_tape_compare", False),
        "hybrid_compact_root_tape_compare",
    )
    hybrid_compact_root_tape_max_records = _positive_int(
        config.get("hybrid_compact_root_tape_max_records", 4),
        "hybrid_compact_root_tape_max_records",
    )
    hybrid_compact_root_tape_allow_resident_host_snapshot = _bool_config(
        config.get("hybrid_compact_root_tape_allow_resident_host_snapshot", False),
        "hybrid_compact_root_tape_allow_resident_host_snapshot",
    )
    hybrid_compact_root_tape_compare_fixed_shape_floor = _bool_config(
        config.get("hybrid_compact_root_tape_compare_fixed_shape_floor", True),
        "hybrid_compact_root_tape_compare_fixed_shape_floor",
    )
    hybrid_compact_root_tape_compare_mctx = _bool_config(
        config.get("hybrid_compact_root_tape_compare_mctx", False),
        "hybrid_compact_root_tape_compare_mctx",
    )
    hybrid_compact_root_tape_compare_model_compile = _bool_config(
        config.get("hybrid_compact_root_tape_compare_model_compile", False),
        "hybrid_compact_root_tape_compare_model_compile",
    )
    hybrid_compact_root_tape_compare_direct_core = _bool_config(
        config.get("hybrid_compact_root_tape_compare_direct_core", False),
        "hybrid_compact_root_tape_compare_direct_core",
    )
    hybrid_compact_root_tape_model_compile_mode = str(
        config.get("hybrid_compact_root_tape_model_compile_mode", "default")
    )
    if hybrid_compact_root_tape_model_compile_mode not in COMPACT_TORCH_MODEL_COMPILE_MODES:
        allowed = ", ".join(COMPACT_TORCH_MODEL_COMPILE_MODES)
        raise ValueError(
            "hybrid_compact_root_tape_model_compile_mode must be one of "
            f"{allowed}; got {hybrid_compact_root_tape_model_compile_mode!r}"
        )
    hybrid_compact_root_tape_require_model_compile = _bool_config(
        config.get("hybrid_compact_root_tape_require_model_compile", True),
        "hybrid_compact_root_tape_require_model_compile",
    )
    hybrid_compact_root_tape_reference_label = str(
        config.get("hybrid_compact_root_tape_reference_label", "primary")
    ).strip()
    if not hybrid_compact_root_tape_reference_label:
        raise ValueError("hybrid_compact_root_tape_reference_label must be non-empty")
    if hybrid_compact_root_tape_compare and not hybrid_observation_canary:
        raise ValueError("hybrid_compact_root_tape_compare requires hybrid_observation_canary")
    if hybrid_compact_root_tape_compare and not hybrid_compact_rollout_slab_probe:
        raise ValueError(
            "hybrid_compact_root_tape_compare requires hybrid_compact_rollout_slab_probe"
        )
    if (
        hybrid_compact_root_tape_compare
        and not hybrid_compact_root_tape_compare_fixed_shape_floor
        and not hybrid_compact_root_tape_compare_mctx
        and not hybrid_compact_root_tape_compare_model_compile
        and not hybrid_compact_root_tape_compare_direct_core
    ):
        raise ValueError(
            "hybrid_compact_root_tape_compare currently requires "
            "at least one secondary service: fixed_shape_floor, MCTX, or "
            "model_compile/direct_core"
        )
    if hybrid_compact_root_tape_compare_mctx and not hybrid_compact_root_tape_compare:
        raise ValueError(
            "hybrid_compact_root_tape_compare_mctx requires hybrid_compact_root_tape_compare"
        )
    if hybrid_compact_root_tape_compare_model_compile and not hybrid_compact_root_tape_compare:
        raise ValueError(
            "hybrid_compact_root_tape_compare_model_compile requires "
            "hybrid_compact_root_tape_compare"
        )
    if hybrid_compact_root_tape_compare_direct_core and not hybrid_compact_root_tape_compare:
        raise ValueError(
            "hybrid_compact_root_tape_compare_direct_core requires hybrid_compact_root_tape_compare"
        )
    if hybrid_compact_root_tape_compare_mctx and hybrid_mctx_compact_search_probe:
        raise ValueError("hybrid_compact_root_tape_compare_mctx is for non-MCTX primary rows")
    if hybrid_compact_root_tape_compare_model_compile and not (
        hybrid_lightzero_array_ceiling_probe
        and str(config.get("hybrid_lightzero_array_ceiling_mode", ""))
        == LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
    ):
        raise ValueError(
            "hybrid_compact_root_tape_compare_model_compile requires "
            "compact_torch_search_service as the LightZero array-ceiling mode"
        )
    if hybrid_compact_root_tape_compare_direct_core and not (
        hybrid_lightzero_array_ceiling_probe
        and str(config.get("hybrid_lightzero_array_ceiling_mode", ""))
        == LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
    ):
        raise ValueError(
            "hybrid_compact_root_tape_compare_direct_core requires "
            "compact_torch_search_service as the LightZero array-ceiling mode"
        )
    if hybrid_compact_root_tape_compare_model_compile:
        if _bool_config(
            config.get("hybrid_compact_torch_compile_model_inference", False),
            "hybrid_compact_torch_compile_model_inference",
        ):
            raise ValueError(
                "hybrid_compact_root_tape_compare_model_compile expects an eager "
                "primary service; leave hybrid_compact_torch_compile_model_inference "
                "false"
            )
        try:
            root_noise_for_model_compile_compare = float(
                config.get("hybrid_lightzero_consumer_root_noise_weight", -1.0)
            )
        except (TypeError, ValueError):
            root_noise_for_model_compile_compare = -1.0
        if root_noise_for_model_compile_compare != 0.0:
            raise ValueError(
                "hybrid_compact_root_tape_compare_model_compile requires "
                "hybrid_lightzero_consumer_root_noise_weight=0.0"
            )
    if hybrid_compact_root_tape_compare_direct_core:
        if _bool_config(
            config.get("hybrid_compact_torch_compile_model_inference", False),
            "hybrid_compact_torch_compile_model_inference",
        ):
            raise ValueError(
                "hybrid_compact_root_tape_compare_direct_core expects model compile off"
            )
        primary_initial_mode = str(
            config.get(
                "hybrid_compact_torch_initial_inference_mode",
                COMPACT_TORCH_INITIAL_INFERENCE_MODE_MODEL_METHOD,
            )
        )
        if primary_initial_mode != COMPACT_TORCH_INITIAL_INFERENCE_MODE_MODEL_METHOD:
            raise ValueError(
                "hybrid_compact_root_tape_compare_direct_core expects a "
                "model_method primary service"
            )
        try:
            root_noise_for_direct_core_compare = float(
                config.get("hybrid_lightzero_consumer_root_noise_weight", -1.0)
            )
        except (TypeError, ValueError):
            root_noise_for_direct_core_compare = -1.0
        if root_noise_for_direct_core_compare != 0.0:
            raise ValueError(
                "hybrid_compact_root_tape_compare_direct_core requires "
                "hybrid_lightzero_consumer_root_noise_weight=0.0"
            )
    if hybrid_compact_root_tape_compare and _bool_config(
        config.get("hybrid_resident_observation_search", False),
        "hybrid_resident_observation_search",
    ):
        raise ValueError(
            "hybrid_compact_root_tape_compare does not yet support resident "
            "observation search; it needs a real explicit device-to-host root "
            "snapshot first"
        )
    hybrid_compact_rollout_slab_learner_gate_impl = str(
        config.get(
            "hybrid_compact_rollout_slab_learner_gate_impl",
            COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_TOY_PROBE,
        )
    )
    if hybrid_compact_rollout_slab_learner_gate_impl not in COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPLS:
        allowed = ", ".join(COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPLS)
        raise ValueError(
            "hybrid_compact_rollout_slab_learner_gate_impl must be one of "
            f"{allowed}; got {hybrid_compact_rollout_slab_learner_gate_impl!r}"
        )
    hybrid_compact_rollout_slab_learner_gate_support_scale = _positive_int(
        config.get("hybrid_compact_rollout_slab_learner_gate_support_scale", 1),
        "hybrid_compact_rollout_slab_learner_gate_support_scale",
    )
    hybrid_compact_rollout_slab_learner_gate_num_unroll_steps = _positive_int(
        config.get("hybrid_compact_rollout_slab_learner_gate_num_unroll_steps", 1),
        "hybrid_compact_rollout_slab_learner_gate_num_unroll_steps",
    )
    if (
        hybrid_compact_rollout_slab_learner_gate_num_unroll_steps != 1
        and hybrid_compact_rollout_slab_learner_gate_impl
        != COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO
    ):
        raise ValueError(
            "hybrid_compact_rollout_slab_learner_gate_num_unroll_steps > 1 "
            "requires compact_muzero learner gate"
        )
    hybrid_compact_owned_loop_entrypoint = _bool_config(
        config.get("hybrid_compact_owned_loop_entrypoint", False),
        "hybrid_compact_owned_loop_entrypoint",
    )
    hybrid_compact_owned_loop_policy_version_ref = str(
        config.get("hybrid_compact_owned_loop_policy_version_ref", "")
    ).strip()
    hybrid_compact_owned_loop_model_version_ref = str(
        config.get("hybrid_compact_owned_loop_model_version_ref", "")
    ).strip()
    hybrid_compact_owned_loop_policy_source = str(
        config.get("hybrid_compact_owned_loop_policy_source", "")
    ).strip()
    hybrid_compact_owned_loop_capture_replay_store_state = _bool_config(
        config.get("hybrid_compact_owned_loop_capture_replay_store_state", False),
        "hybrid_compact_owned_loop_capture_replay_store_state",
    )
    if hybrid_compact_owned_loop_entrypoint:
        if not hybrid_compact_rollout_slab_probe:
            raise ValueError(
                "hybrid_compact_owned_loop_entrypoint requires hybrid_compact_rollout_slab_probe"
            )
        if not _bool_config(
            config.get("hybrid_compact_rollout_slab_sample_gate", False),
            "hybrid_compact_rollout_slab_sample_gate",
        ):
            raise ValueError(
                "hybrid_compact_owned_loop_entrypoint requires "
                "hybrid_compact_rollout_slab_sample_gate"
            )
        if not _bool_config(
            config.get("hybrid_compact_rollout_slab_learner_gate", False),
            "hybrid_compact_rollout_slab_learner_gate",
        ):
            raise ValueError(
                "hybrid_compact_owned_loop_entrypoint requires "
                "hybrid_compact_rollout_slab_learner_gate"
            )
        if _bool_config(
            config.get("hybrid_materialize_scalar_timestep", True),
            "hybrid_materialize_scalar_timestep",
        ):
            raise ValueError(
                "hybrid_compact_owned_loop_entrypoint requires "
                "hybrid_materialize_scalar_timestep=false"
            )
        if not hybrid_compact_owned_loop_policy_version_ref:
            raise ValueError("hybrid_compact_owned_loop_policy_version_ref must be non-empty")
        if not hybrid_compact_owned_loop_policy_source:
            raise ValueError("hybrid_compact_owned_loop_policy_source must be non-empty")
    hybrid_device_only_stack = _bool_config(
        config.get("hybrid_device_only_stack", False),
        "hybrid_device_only_stack",
    )
    hybrid_refresh_observation_stack = _bool_config(
        config.get("hybrid_refresh_observation_stack", True),
        "hybrid_refresh_observation_stack",
    )
    hybrid_resident_observation_search = _bool_config(
        config.get("hybrid_resident_observation_search", False),
        "hybrid_resident_observation_search",
    )
    hybrid_native_actor_buffer = _bool_config(
        config.get("hybrid_native_actor_buffer", False),
        "hybrid_native_actor_buffer",
    )
    hybrid_persistent_compact_render_state_buffer = _bool_config(
        config.get("hybrid_persistent_compact_render_state_buffer", False),
        "hybrid_persistent_compact_render_state_buffer",
    )
    hybrid_borrow_single_actor_render_state = _bool_config(
        config.get("hybrid_borrow_single_actor_render_state", False),
        "hybrid_borrow_single_actor_render_state",
    )
    if hybrid_persistent_compact_render_state_buffer and not hybrid_native_actor_buffer:
        raise ValueError(
            "hybrid_persistent_compact_render_state_buffer requires hybrid_native_actor_buffer"
        )
    if hybrid_borrow_single_actor_render_state:
        if not hybrid_native_actor_buffer:
            raise ValueError(
                "hybrid_borrow_single_actor_render_state requires hybrid_native_actor_buffer"
            )
        if int(actor_count) != 1:
            raise ValueError("hybrid_borrow_single_actor_render_state requires actor_count=1")
        if hybrid_persistent_compact_render_state_buffer:
            raise ValueError(
                "hybrid_borrow_single_actor_render_state cannot be combined with "
                "hybrid_persistent_compact_render_state_buffer"
            )
        if not hybrid_refresh_observation_stack:
            raise ValueError(
                "hybrid_borrow_single_actor_render_state requires "
                "hybrid_refresh_observation_stack"
            )
    hybrid_lightzero_mock_service_materialize_public_output = _bool_config(
        config.get("hybrid_lightzero_mock_service_materialize_public_output", False),
        "hybrid_lightzero_mock_service_materialize_public_output",
    )
    hybrid_compact_torch_compile_search = _bool_config(
        config.get("hybrid_compact_torch_compile_search", True),
        "hybrid_compact_torch_compile_search",
    )
    hybrid_compact_torch_compile_model_inference = _bool_config(
        config.get("hybrid_compact_torch_compile_model_inference", False),
        "hybrid_compact_torch_compile_model_inference",
    )
    hybrid_compact_torch_require_model_compile = _bool_config(
        config.get("hybrid_compact_torch_require_model_compile", False),
        "hybrid_compact_torch_require_model_compile",
    )
    hybrid_compact_torch_model_compile_mode = str(
        config.get("hybrid_compact_torch_model_compile_mode", "reduce-overhead")
    )
    if hybrid_compact_torch_model_compile_mode not in COMPACT_TORCH_MODEL_COMPILE_MODES:
        allowed = ", ".join(COMPACT_TORCH_MODEL_COMPILE_MODES)
        raise ValueError(
            "hybrid_compact_torch_model_compile_mode must be one of "
            f"{allowed}; got {hybrid_compact_torch_model_compile_mode!r}"
        )
    hybrid_compact_torch_initial_inference_mode = str(
        config.get("hybrid_compact_torch_initial_inference_mode", "model_method")
    )
    if hybrid_compact_torch_initial_inference_mode not in COMPACT_TORCH_INITIAL_INFERENCE_MODES:
        allowed = ", ".join(COMPACT_TORCH_INITIAL_INFERENCE_MODES)
        raise ValueError(
            "hybrid_compact_torch_initial_inference_mode must be one of "
            f"{allowed}; got {hybrid_compact_torch_initial_inference_mode!r}"
        )
    hybrid_compact_torch_observation_memory_format = str(
        config.get("hybrid_compact_torch_observation_memory_format", "contiguous")
    )
    if hybrid_compact_torch_observation_memory_format not in COMPACT_TORCH_MEMORY_FORMATS:
        allowed = ", ".join(COMPACT_TORCH_MEMORY_FORMATS)
        raise ValueError(
            "hybrid_compact_torch_observation_memory_format must be one of "
            f"{allowed}; got {hybrid_compact_torch_observation_memory_format!r}"
        )
    hybrid_compact_torch_model_memory_format = str(
        config.get("hybrid_compact_torch_model_memory_format", "contiguous")
    )
    if hybrid_compact_torch_model_memory_format not in COMPACT_TORCH_MEMORY_FORMATS:
        allowed = ", ".join(COMPACT_TORCH_MEMORY_FORMATS)
        raise ValueError(
            "hybrid_compact_torch_model_memory_format must be one of "
            f"{allowed}; got {hybrid_compact_torch_model_memory_format!r}"
        )
    if hybrid_compact_torch_model_memory_format != "contiguous":
        raise ValueError(
            "hybrid_compact_torch_model_memory_format=channels_last is parked for the "
            "current LightZero MuZero model because recurrent dynamics uses .view(); "
            "use hybrid_compact_torch_model_memory_format='contiguous'"
        )
    if (
        hybrid_lightzero_mock_service_materialize_public_output
        and hybrid_lightzero_array_ceiling_mode != LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE
    ):
        raise ValueError(
            "hybrid_lightzero_mock_service_materialize_public_output requires "
            "hybrid_lightzero_array_ceiling_mode='mock_search_service'"
        )
    if render_surface == RENDER_SURFACE_DIRECT_GRAY64:
        if not (surface_facade_canary or profile_env_manager_canary or hybrid_observation_canary):
            raise ValueError(
                "direct_gray64 render_surface is only supported for "
                "surface_facade_canary, profile_env_manager_canary, or "
                "hybrid_observation_canary"
            )
        if surface_stack_backend != TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE:
            raise ValueError(
                "direct_gray64 render_surface requires "
                f"surface_stack_backend={TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE!r}"
            )
    if (
        observation_renderer_backend
        == SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
    ):
        if not (surface_facade_canary or profile_env_manager_canary or hybrid_observation_canary):
            raise ValueError(
                "jax_gpu_persistent_policy_framebuffer_profile is only supported for "
                "surface_facade_canary, profile_env_manager_canary, or "
                "hybrid_observation_canary profile rows"
            )
        if surface_stack_backend != TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE:
            raise ValueError(
                "jax_gpu_persistent_policy_framebuffer_profile requires "
                f"surface_stack_backend={TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE!r}"
            )
        if render_surface != RENDER_SURFACE_DIRECT_GRAY64:
            raise ValueError(
                "jax_gpu_persistent_policy_framebuffer_profile currently renders the "
                "policy-space direct_gray64 surface; set render_surface='direct_gray64'"
            )
    elif async_device_only_profile:
        raise ValueError(
            "async_device_only_profile requires "
            f"observation_renderer_backend="
            f"{SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND!r}"
        )
    if (
        hybrid_lightzero_collect_forward_probe
        or hybrid_lightzero_initial_inference_probe
        or hybrid_lightzero_array_ceiling_probe
        or hybrid_lightzero_mcts_arrays_boundary_probe
        or hybrid_mctx_compact_search_probe
    ):
        probe_name = (
            "hybrid_lightzero_collect_forward_probe"
            if hybrid_lightzero_collect_forward_probe
            else (
                "hybrid_lightzero_initial_inference_probe"
                if hybrid_lightzero_initial_inference_probe
                else (
                    "hybrid_lightzero_array_ceiling_probe"
                    if hybrid_lightzero_array_ceiling_probe
                    else (
                        "hybrid_lightzero_mcts_arrays_boundary_probe"
                        if hybrid_lightzero_mcts_arrays_boundary_probe
                        else "hybrid_mctx_compact_search_probe"
                    )
                )
            )
        )
        if not hybrid_observation_canary:
            raise ValueError(f"{probe_name} requires hybrid_observation_canary")
        if (
            observation_renderer_backend
            != SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
        ):
            raise ValueError(
                f"{probe_name} requires "
                f"observation_renderer_backend="
                f"{SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND!r}"
            )
        if render_surface != RENDER_SURFACE_DIRECT_GRAY64:
            raise ValueError(
                f"{probe_name} requires render_surface={RENDER_SURFACE_DIRECT_GRAY64!r}"
            )
    allow_render_truncation = _bool_config(
        config.get("allow_render_truncation", False),
        "allow_render_truncation",
    )
    prewarm_render_functions = _bool_config(
        config.get("prewarm_render_functions", False),
        "prewarm_render_functions",
    )
    surface_facade_divergence_canary = _bool_config(
        config.get("surface_facade_divergence_canary", False),
        "surface_facade_divergence_canary",
    )
    if surface_facade_divergence_canary and not surface_facade_canary:
        raise ValueError("surface_facade_divergence_canary requires surface_facade_canary")
    dynamic_render_trail_slots = _bool_config(
        config.get("dynamic_render_trail_slots", True),
        "dynamic_render_trail_slots",
    )
    default_min_render_trail_slots = (
        min(DEFAULT_DYNAMIC_MIN_RENDER_TRAIL_SLOTS, trail_slots)
        if dynamic_render_trail_slots
        else trail_slots
    )
    min_render_trail_slots = _positive_int(
        config.get("min_render_trail_slots", default_min_render_trail_slots),
        "min_render_trail_slots",
    )
    if min_render_trail_slots > trail_slots:
        raise ValueError(
            "min_render_trail_slots must be less than or equal to trail_slots; "
            f"got min_render_trail_slots={min_render_trail_slots}, trail_slots={trail_slots}"
        )
    rnd_batch_size = _positive_int(config.get("rnd_batch_size", 64), "rnd_batch_size")
    rnd_update_per_collect = _positive_int(
        config.get("rnd_update_per_collect", xb.RND_DEFAULT_UPDATE_PER_COLLECT),
        "rnd_update_per_collect",
    )
    rnd_device = str(config.get("rnd_device", "cpu"))
    geometry_dtype = str(config.get("geometry_dtype", DEFAULT_BOUNDARY_GEOMETRY_DTYPE))
    if geometry_dtype not in {"float32", "float64"}:
        raise ValueError(f"geometry_dtype must be float32 or float64, got {geometry_dtype!r}")
    requested_parity_mode = str(config.get("parity_mode", DEFAULT_BOUNDARY_PARITY_MODE))
    if requested_parity_mode not in BOUNDARY_PARITY_MODES:
        allowed = ", ".join(BOUNDARY_PARITY_MODES)
        raise ValueError(f"parity_mode must be one of {allowed}; got {requested_parity_mode!r}")
    parity_mode = (
        BOUNDARY_PARITY_MODE_EXACT if geometry_dtype == "float64" else BOUNDARY_PARITY_MODE_TOLERANT
    )
    if requested_parity_mode != BOUNDARY_PARITY_MODE_AUTO:
        parity_mode = requested_parity_mode
    parity_max_abs_diff = _nonnegative_int(
        config.get("parity_max_abs_diff", DEFAULT_BOUNDARY_PARITY_MAX_ABS_DIFF),
        "parity_max_abs_diff",
    )
    parity_max_mismatch_fraction = float(
        config.get(
            "parity_max_mismatch_fraction",
            DEFAULT_BOUNDARY_PARITY_MAX_MISMATCH_FRACTION,
        )
    )
    if parity_max_mismatch_fraction < 0.0:
        raise ValueError("parity_max_mismatch_fraction must be non-negative")
    if include_rnd_meter and not include_lightzero_payload_profile:
        raise ValueError("include_rnd_meter requires include_lightzero_payload_profile")

    probe_env = VectorMultiplayerEnv(
        batch_size,
        player_count=PLAYER_COUNT,
        seed=seed,
        body_capacity=body_capacity,
        natural_bonus_spawn=False,
        death_mode=death_mode,
        max_ticks=max_ticks,
    )
    map_size = float(getattr(probe_env, "map_size"))
    bonus_active = probe_env.state.get("bonus_active")
    bonus_count = 0 if bonus_active is None else int(np.asarray(bonus_active).shape[1])
    render_config = _validate_config(
        {
            "state_source": "real_env_rollout",
            "batch_size": batch_size,
            "player_count": PLAYER_COUNT,
            "controlled_player": 0,
            "trail_slots": trail_slots,
            "render_mode": RENDER_MODE_BROWSER_LINES,
            "bonus_render_mode": BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
            "render_surface": render_surface,
            "trail_composition": (
                TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT
                if render_surface == RENDER_SURFACE_BLOCK_704_GRAY64
                else TRAIL_COMPOSITION_PRIORITY_BUFFER
            ),
            "render_views": RENDER_VIEWS_BOTH,
            "geometry_dtype": geometry_dtype,
            "bonus_count": bonus_count,
            "frame_size": 704,
            "target_size": TARGET_SIZE,
            "seed": seed,
            "warmup_runs": warmup_steps,
            "steady_runs": steps,
            "real_env_steps": 0,
            "verify_rows": batch_size,
            "transfer_output": True,
            "map_size": map_size,
        }
    )
    render_config["body_capacity"] = body_capacity
    render_config["allow_render_truncation"] = allow_render_truncation
    render_config["dynamic_render_trail_slots"] = dynamic_render_trail_slots
    render_config["min_render_trail_slots"] = min_render_trail_slots
    render_config["max_render_trail_slots"] = trail_slots
    render_config["async_device_only_profile"] = async_device_only_profile
    return {
        "batch_size": batch_size,
        "actor_count": actor_count,
        "compute": compute,
        "player_count": PLAYER_COUNT,
        "death_mode": death_mode,
        "seed": seed,
        "trail_slots": trail_slots,
        "body_capacity": body_capacity,
        "dynamic_render_trail_slots": dynamic_render_trail_slots,
        "min_render_trail_slots": min_render_trail_slots,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "verify_steps": verify_steps,
        "cpu_reference_interval": cpu_reference_interval,
        "max_ticks": max_ticks,
        "include_lightzero_payload_profile": include_lightzero_payload_profile,
        "pickle_lightzero_payload": pickle_lightzero_payload,
        "include_rnd_meter": include_rnd_meter,
        "surface_facade_canary": surface_facade_canary,
        "surface_facade_divergence_canary": surface_facade_divergence_canary,
        "profile_env_manager_canary": profile_env_manager_canary,
        "hybrid_observation_canary": hybrid_observation_canary,
        "hybrid_lightzero_collect_forward_probe": hybrid_lightzero_collect_forward_probe,
        "hybrid_lightzero_initial_inference_probe": hybrid_lightzero_initial_inference_probe,
        "hybrid_lightzero_array_ceiling_probe": hybrid_lightzero_array_ceiling_probe,
        "hybrid_lightzero_mcts_arrays_boundary_probe": (
            hybrid_lightzero_mcts_arrays_boundary_probe
        ),
        "hybrid_mctx_compact_search_probe": hybrid_mctx_compact_search_probe,
        "hybrid_mctx_lightzero_checkpoint_ref": hybrid_mctx_lightzero_checkpoint_ref,
        "hybrid_mctx_lightzero_checkpoint_state_key": hybrid_mctx_lightzero_checkpoint_state_key,
        "hybrid_mctx_compare_direct_ctree": hybrid_mctx_compare_direct_ctree,
        "hybrid_mctx_compare_direct_ctree_impl": hybrid_mctx_compare_direct_ctree_impl,
        "hybrid_lightzero_mcts_arrays_boundary_impl": (hybrid_lightzero_mcts_arrays_boundary_impl),
        "hybrid_lightzero_mcts_arrays_boundary_input_mode": (
            hybrid_lightzero_mcts_arrays_boundary_input_mode
        ),
        "hybrid_lightzero_array_ceiling_mode": hybrid_lightzero_array_ceiling_mode,
        "hybrid_lightzero_array_ceiling_input_mode": (hybrid_lightzero_array_ceiling_input_mode),
        "hybrid_lightzero_mock_service_materialize_public_output": (
            hybrid_lightzero_mock_service_materialize_public_output
        ),
        "hybrid_compact_torch_compile_search": hybrid_compact_torch_compile_search,
        "hybrid_compact_torch_compile_model_inference": (
            hybrid_compact_torch_compile_model_inference
        ),
        "hybrid_compact_torch_require_model_compile": (hybrid_compact_torch_require_model_compile),
        "hybrid_compact_torch_model_compile_mode": (hybrid_compact_torch_model_compile_mode),
        "hybrid_compact_torch_initial_inference_mode": (
            hybrid_compact_torch_initial_inference_mode
        ),
        "hybrid_compact_torch_observation_memory_format": (
            hybrid_compact_torch_observation_memory_format
        ),
        "hybrid_compact_torch_model_memory_format": (hybrid_compact_torch_model_memory_format),
        "hybrid_compact_rollout_slab_action_mode": hybrid_compact_rollout_slab_action_mode,
        "hybrid_compact_root_tape_compare": hybrid_compact_root_tape_compare,
        "hybrid_compact_root_tape_max_records": hybrid_compact_root_tape_max_records,
        "hybrid_compact_root_tape_allow_resident_host_snapshot": (
            hybrid_compact_root_tape_allow_resident_host_snapshot
        ),
        "hybrid_compact_root_tape_compare_fixed_shape_floor": (
            hybrid_compact_root_tape_compare_fixed_shape_floor
        ),
        "hybrid_compact_root_tape_compare_mctx": (hybrid_compact_root_tape_compare_mctx),
        "hybrid_compact_root_tape_compare_model_compile": (
            hybrid_compact_root_tape_compare_model_compile
        ),
        "hybrid_compact_root_tape_compare_direct_core": (
            hybrid_compact_root_tape_compare_direct_core
        ),
        "hybrid_compact_root_tape_model_compile_mode": (
            hybrid_compact_root_tape_model_compile_mode
        ),
        "hybrid_compact_root_tape_require_model_compile": (
            hybrid_compact_root_tape_require_model_compile
        ),
        "hybrid_compact_root_tape_reference_label": (hybrid_compact_root_tape_reference_label),
        "hybrid_compact_rollout_slab_learner_gate_impl": (
            hybrid_compact_rollout_slab_learner_gate_impl
        ),
        "hybrid_compact_rollout_slab_learner_gate_support_scale": (
            hybrid_compact_rollout_slab_learner_gate_support_scale
        ),
        "hybrid_compact_rollout_slab_learner_gate_num_unroll_steps": (
            hybrid_compact_rollout_slab_learner_gate_num_unroll_steps
        ),
        "hybrid_compact_owned_loop_entrypoint": (hybrid_compact_owned_loop_entrypoint),
        "hybrid_compact_owned_loop_policy_version_ref": (
            hybrid_compact_owned_loop_policy_version_ref
        ),
        "hybrid_compact_owned_loop_model_version_ref": (
            hybrid_compact_owned_loop_model_version_ref
        ),
        "hybrid_compact_owned_loop_policy_source": (hybrid_compact_owned_loop_policy_source),
        "hybrid_compact_owned_loop_capture_replay_store_state": (
            hybrid_compact_owned_loop_capture_replay_store_state
        ),
        "hybrid_device_only_stack": hybrid_device_only_stack,
        "hybrid_refresh_observation_stack": hybrid_refresh_observation_stack,
        "hybrid_resident_observation_search": hybrid_resident_observation_search,
        "hybrid_native_actor_buffer": hybrid_native_actor_buffer,
        "hybrid_persistent_compact_render_state_buffer": (
            hybrid_persistent_compact_render_state_buffer
        ),
        "hybrid_borrow_single_actor_render_state": hybrid_borrow_single_actor_render_state,
        "hybrid_mctx_num_simulations": _positive_int(
            config.get("hybrid_mctx_num_simulations", 8),
            "hybrid_mctx_num_simulations",
        ),
        "hybrid_mctx_hidden_dim": _positive_int(
            config.get("hybrid_mctx_hidden_dim", 64),
            "hybrid_mctx_hidden_dim",
        ),
        "hybrid_mctx_visual_channels": _positive_int(
            config.get("hybrid_mctx_visual_channels", 8),
            "hybrid_mctx_visual_channels",
        ),
        "hybrid_mctx_require_gpu_backend": _bool_config(
            config.get("hybrid_mctx_require_gpu_backend", True),
            "hybrid_mctx_require_gpu_backend",
        ),
        "surface_stack_backend": surface_stack_backend,
        "observation_renderer_backend": observation_renderer_backend,
        "async_device_only_profile": async_device_only_profile,
        "persistent_vectorized_delta_pack_profile": bool(
            config.get("persistent_vectorized_delta_pack_profile", False)
        ),
        "render_surface": render_surface,
        "allow_render_truncation": allow_render_truncation,
        "prewarm_render_functions": prewarm_render_functions,
        "rnd_batch_size": rnd_batch_size,
        "rnd_update_per_collect": rnd_update_per_collect,
        "rnd_device": rnd_device,
        "geometry_dtype": geometry_dtype,
        "requested_parity_mode": requested_parity_mode,
        "parity_mode": parity_mode,
        "parity_max_abs_diff": parity_max_abs_diff,
        "parity_max_mismatch_fraction": parity_max_mismatch_fraction,
        "map_size": map_size,
        "render_config": render_config,
    }


def _run_surface_facade_profile_impl(config: dict[str, Any]) -> dict[str, Any]:
    import jax
    import numpy as np

    if str(config.get("geometry_dtype", DEFAULT_BOUNDARY_GEOMETRY_DTYPE)) == "float64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    checked = _validate_boundary_config(np=np, config=config)
    render_mode_id = RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]
    bonus_render_mode_id = BONUS_RENDER_MODE_IDS[BONUS_RENDER_MODE_SIMPLE_SYMBOLS]
    render_fn_cache: dict[int, Any] = {}

    def render_fn_for_slots(render_trail_slots: int) -> Any:
        slots = int(render_trail_slots)
        cached = render_fn_cache.get(slots)
        if cached is not None:
            return cached
        slot_config = dict(checked["render_config"])
        slot_config["trail_slots"] = slots
        cached = _make_jax_two_view_render_fn(
            jax=jax,
            jnp=jnp,
            config=slot_config,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
        )
        render_fn_cache[slots] = cached
        return cached

    surface_stack_backend = str(checked["surface_stack_backend"])
    surface_kwargs: dict[str, Any] = {
        "batch_size": checked["batch_size"],
        "player_count": PLAYER_COUNT,
        "seed": checked["seed"],
        "decision_source_frames": 1,
        "source_physics_step_ms": SOURCE_PHYSICS_STEP_MS,
        "body_capacity": checked["body_capacity"],
        "map_size": checked["map_size"],
        "natural_bonus_spawn": False,
        "death_mode": checked["death_mode"],
        "max_ticks": checked["max_ticks"],
        "observation_stack_backend": surface_stack_backend,
    }
    if surface_stack_backend == TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE:
        renderer = _make_profile_observation_renderer(
            jax=jax,
            jnp=jnp,
            np=np,
            checked=checked,
            render_fn_for_slots=render_fn_for_slots,
            bonus_render_mode_id=bonus_render_mode_id,
        )
        surface_kwargs["observation_renderer"] = renderer
        surface_kwargs["required_observation_renderer_backend"] = renderer.backend_name
    surface = SourceStateMultiplayerTrainerSurface(**surface_kwargs)
    rng = np.random.default_rng(int(checked["seed"]))
    rnd_model = _build_rnd_model(checked) if checked["include_rnd_meter"] else None
    divergence_enabled = bool(checked["surface_facade_divergence_canary"])
    cpu_renderer = CpuOracleBatchedObservationRenderer() if divergence_enabled else None
    reference_stacks = (
        np.zeros(
            (
                int(checked["batch_size"]),
                PLAYER_COUNT,
                POLICY_FRAME_STACK_DEPTH,
                TARGET_SIZE,
                TARGET_SIZE,
            ),
            dtype=np.float32,
        )
        if divergence_enabled
        else None
    )
    reference_render_rows = (
        _row_major_render_rows(np=np, batch_size=checked["batch_size"])
        if divergence_enabled
        else None
    )
    reference_render_players = (
        _row_major_render_players(np=np, batch_size=checked["batch_size"])
        if divergence_enabled
        else None
    )
    reference_raw_out = (
        np.zeros(
            (
                checked["batch_size"] * PLAYER_COUNT,
                1,
                TARGET_SIZE,
                TARGET_SIZE,
            ),
            dtype=np.uint8,
        )
        if divergence_enabled
        else None
    )
    divergence_summaries: list[dict[str, Any]] = []
    divergence_checks = 0
    divergence_timings = {field: [] for field in SURFACE_FACADE_DIVERGENCE_TIMING_FIELDS}

    reset_started = time.perf_counter()
    reset_step = surface.reset(seed=checked["seed"])
    reset_total_sec = time.perf_counter() - reset_started
    if divergence_enabled:
        if (
            cpu_renderer is None
            or reference_stacks is None
            or reference_render_rows is None
            or reference_render_players is None
            or reference_raw_out is None
        ):
            raise RuntimeError("internal error: divergence reference state was not initialized")
        reset_reference_frames, reset_reference_sec = _render_cpu_reference_frames(
            renderer=cpu_renderer,
            env=surface.env,
            row_indices=reference_render_rows,
            controlled_players=reference_render_players,
            out=reference_raw_out,
            reference_stacks=reference_stacks,
        )
        divergence_timings["candidate_reset_sec"].append(reset_total_sec)
        divergence_timings["cpu_reference_reset_render_stack_sec"].append(reset_reference_sec)
        divergence_summaries.append(
            _assert_parity(
                label="surface_reset",
                candidate_frames=_latest_uint8_frames_from_stack(
                    np=np, stack=reset_step.observation
                ),
                reference_frames=reset_reference_frames,
                candidate_stacks=reset_step.observation,
                reference_stacks=reference_stacks,
                config=checked,
            )
        )

    per_step: dict[str, list[float]] = {
        "surface_step_total_sec": [],
        "surface_renderer_render_sec": [],
        "surface_renderer_device_render_sec": [],
        "surface_renderer_host_to_device_sec": [],
        "surface_renderer_device_to_host_sec": [],
        "surface_renderer_pack_sec": [],
        "surface_nonrenderer_sec": [],
        "surface_env_step_sec": [],
        "surface_stack_update_sec": [],
        "surface_reward_sec": [],
        "surface_package_sec": [],
        "lightzero_scalarize_sec": [],
        "lightzero_payload_pickle_sec": [],
        "lightzero_payload_total_sec": [],
        "rnd_collect_data_sec": [],
        "rnd_train_with_data_sec": [],
        "rnd_estimate_sec": [],
        "surface_autoreset_sec": [],
    }
    numeric: dict[str, list[float]] = {
        "policy_row_count": [],
        "terminal_row_count": [],
        "final_observation_row_count": [],
        "render_trail_slots": [],
        "active_trail_count_max": [],
        "render_truncation_row_count": [],
        "persistent_reset_row_count": [],
        "persistent_delta_slot_count": [],
        "persistent_delta_max_slots": [],
        "persistent_update_fn_cache_size": [],
        "persistent_partial_render_request": [],
        "lightzero_payload_pickle_bytes": [],
    }

    total_iterations = int(checked["warmup_steps"]) + int(checked["steps"])
    for iteration in range(total_iterations):
        joint_action = rng.integers(
            0,
            ACTION_COUNT,
            size=(int(checked["batch_size"]), PLAYER_COUNT),
            dtype=np.int16,
        )
        started = time.perf_counter()
        step = surface.step(joint_action)
        surface_step_total_sec = time.perf_counter() - started
        terminal_rows = np.asarray(step.done, dtype=bool)
        reference_frames = None
        reference_render_sec = 0.0
        if divergence_enabled and divergence_checks < int(checked["verify_steps"]):
            if (
                cpu_renderer is None
                or reference_stacks is None
                or reference_render_rows is None
                or reference_render_players is None
                or reference_raw_out is None
            ):
                raise RuntimeError("internal error: divergence reference state was not initialized")
            reference_frames, reference_render_sec = _render_cpu_reference_frames(
                renderer=cpu_renderer,
                env=surface.env,
                row_indices=reference_render_rows,
                controlled_players=reference_render_players,
                out=reference_raw_out,
                reference_stacks=reference_stacks,
            )
        surface_autoreset_sec = 0.0
        if iteration < int(checked["warmup_steps"]):
            if bool(terminal_rows.any()):
                started = time.perf_counter()
                reset_after_terminal = surface.reset(row_mask=terminal_rows)
                surface_autoreset_sec = time.perf_counter() - started
                if divergence_enabled and divergence_checks < int(checked["verify_steps"]):
                    if (
                        cpu_renderer is None
                        or reference_stacks is None
                        or reference_render_rows is None
                        or reference_render_players is None
                        or reference_raw_out is None
                    ):
                        raise RuntimeError(
                            "internal error: divergence reference state was not initialized"
                        )
                    reset_reference_frames, reset_reference_sec = _render_cpu_reference_frames(
                        renderer=cpu_renderer,
                        env=surface.env,
                        row_indices=reference_render_rows,
                        controlled_players=reference_render_players,
                        out=reference_raw_out,
                        reference_stacks=reference_stacks,
                        row_mask=terminal_rows,
                        reset_selected_rows=True,
                    )
                    divergence_timings["candidate_reset_sec"].append(surface_autoreset_sec)
                    divergence_timings["cpu_reference_reset_render_stack_sec"].append(
                        reset_reference_sec
                    )
                    divergence_summaries.append(
                        _assert_parity(
                            label=f"surface_warmup_autoreset_{iteration}",
                            candidate_frames=_latest_uint8_frames_from_stack(
                                np=np,
                                stack=reset_after_terminal.observation,
                            ),
                            reference_frames=reset_reference_frames,
                            candidate_stacks=reset_after_terminal.observation,
                            reference_stacks=reference_stacks,
                            config=checked,
                        )
                    )
            continue

        if divergence_enabled and divergence_checks < int(checked["verify_steps"]):
            if reference_frames is None or reference_stacks is None:
                raise RuntimeError("internal error: divergence reference was not rendered")
            divergence_summaries.append(
                _assert_parity(
                    label=f"surface_step_{iteration - int(checked['warmup_steps'])}",
                    candidate_frames=_latest_uint8_frames_from_stack(
                        np=np,
                        stack=step.observation,
                    ),
                    reference_frames=reference_frames,
                    candidate_stacks=step.observation,
                    reference_stacks=reference_stacks,
                    config=checked,
                )
            )
            if bool(terminal_rows.any()):
                divergence_summaries.append(
                    _assert_final_observation_parity(
                        label=f"surface_terminal_{iteration - int(checked['warmup_steps'])}",
                        candidate_final_observation=step.final_observation,
                        reference_final_observation=reference_stacks,
                        done_mask=terminal_rows,
                        config=checked,
                    )
                )
            divergence_timings["candidate_surface_step_sec"].append(surface_step_total_sec)
            divergence_timings["cpu_reference_render_stack_sec"].append(reference_render_sec)
            divergence_checks += 1

        telemetry = dict(step.info.get("renderer_backed_stack_telemetry") or {})
        surface_timing = dict(step.info.get("trainer_surface_profile_timing") or {})
        lightzero_scalarize_sec = 0.0
        lightzero_payload_pickle_sec = 0.0
        lightzero_payload_pickle_bytes = 0
        rnd_collect_data_sec = 0.0
        rnd_train_with_data_sec = 0.0
        rnd_estimate_sec = 0.0
        if checked["include_lightzero_payload_profile"]:
            started = time.perf_counter()
            timestep, flat_obs, target_reward = materialize_trainer_surface_policy_timestep(
                surface_step=step,
                batch_size=checked["batch_size"],
                player_count=PLAYER_COUNT,
            )
            lightzero_scalarize_sec = time.perf_counter() - started
            if checked["pickle_lightzero_payload"]:
                started = time.perf_counter()
                payload = pickle.dumps(timestep, protocol=pickle.HIGHEST_PROTOCOL)
                lightzero_payload_pickle_sec = time.perf_counter() - started
                lightzero_payload_pickle_bytes = len(payload)
            if rnd_model is not None and flat_obs.shape[0] > 0:
                segment = SimpleNamespace(obs_segment=flat_obs)
                started = time.perf_counter()
                rnd_model.collect_data([[segment]])
                rnd_collect_data_sec = time.perf_counter() - started

                started = time.perf_counter()
                rnd_model.train_with_data()
                rnd_train_with_data_sec = time.perf_counter() - started

                started = time.perf_counter()
                rnd_model.estimate([[flat_obs], [target_reward]])
                rnd_estimate_sec = time.perf_counter() - started

        if bool(terminal_rows.any()):
            started = time.perf_counter()
            reset_after_terminal = surface.reset(row_mask=terminal_rows)
            surface_autoreset_sec = time.perf_counter() - started
            if divergence_enabled and divergence_checks < int(checked["verify_steps"]):
                if (
                    cpu_renderer is None
                    or reference_stacks is None
                    or reference_render_rows is None
                    or reference_render_players is None
                    or reference_raw_out is None
                ):
                    raise RuntimeError(
                        "internal error: divergence reference state was not initialized"
                    )
                reset_reference_frames, reset_reference_sec = _render_cpu_reference_frames(
                    renderer=cpu_renderer,
                    env=surface.env,
                    row_indices=reference_render_rows,
                    controlled_players=reference_render_players,
                    out=reference_raw_out,
                    reference_stacks=reference_stacks,
                    row_mask=terminal_rows,
                    reset_selected_rows=True,
                )
                divergence_timings["candidate_reset_sec"].append(surface_autoreset_sec)
                divergence_timings["cpu_reference_reset_render_stack_sec"].append(
                    reset_reference_sec
                )
                divergence_summaries.append(
                    _assert_parity(
                        label=f"surface_autoreset_{iteration - int(checked['warmup_steps'])}",
                        candidate_frames=_latest_uint8_frames_from_stack(
                            np=np,
                            stack=reset_after_terminal.observation,
                        ),
                        reference_frames=reset_reference_frames,
                        candidate_stacks=reset_after_terminal.observation,
                        reference_stacks=reference_stacks,
                        config=checked,
                    )
                )

        per_step["surface_step_total_sec"].append(surface_step_total_sec)
        per_step["surface_renderer_render_sec"].append(float(telemetry.get("render_sec", 0.0)))
        per_step["surface_renderer_device_render_sec"].append(
            float(telemetry.get("device_render_sec", 0.0))
        )
        per_step["surface_renderer_host_to_device_sec"].append(
            float(telemetry.get("host_to_device_sec", 0.0))
        )
        per_step["surface_renderer_device_to_host_sec"].append(
            float(telemetry.get("device_to_host_sec", 0.0))
        )
        per_step["surface_renderer_pack_sec"].append(
            float(telemetry.get("production_to_compact_sec", 0.0))
            + float(telemetry.get("owner_ordered_pack_sec", 0.0))
        )
        render_total_sec = float(telemetry.get("render_sec", 0.0))
        per_step["surface_nonrenderer_sec"].append(
            max(0.0, surface_step_total_sec - render_total_sec)
        )
        per_step["surface_env_step_sec"].append(float(surface_timing.get("env_step_sec", 0.0)))
        per_step["surface_stack_update_sec"].append(
            float(surface_timing.get("stack_update_sec", 0.0))
        )
        per_step["surface_reward_sec"].append(float(surface_timing.get("reward_sec", 0.0)))
        per_step["surface_package_sec"].append(float(surface_timing.get("package_sec", 0.0)))
        per_step["lightzero_scalarize_sec"].append(lightzero_scalarize_sec)
        per_step["lightzero_payload_pickle_sec"].append(lightzero_payload_pickle_sec)
        per_step["lightzero_payload_total_sec"].append(
            lightzero_scalarize_sec
            + lightzero_payload_pickle_sec
            + rnd_collect_data_sec
            + rnd_train_with_data_sec
            + rnd_estimate_sec
        )
        per_step["rnd_collect_data_sec"].append(rnd_collect_data_sec)
        per_step["rnd_train_with_data_sec"].append(rnd_train_with_data_sec)
        per_step["rnd_estimate_sec"].append(rnd_estimate_sec)
        per_step["surface_autoreset_sec"].append(surface_autoreset_sec)
        numeric["policy_row_count"].append(float(step.info.get("policy_row_count", 0)))
        numeric["terminal_row_count"].append(float(np.count_nonzero(terminal_rows)))
        numeric["final_observation_row_count"].append(
            float(np.count_nonzero(np.asarray(step.final_observation_row_mask, dtype=bool)))
        )
        numeric["render_trail_slots"].append(float(telemetry.get("render_trail_slots", 0.0)))
        numeric["active_trail_count_max"].append(
            float(telemetry.get("active_trail_count_max", 0.0))
        )
        numeric["render_truncation_row_count"].append(
            float(telemetry.get("render_truncation_row_count", 0.0))
        )
        numeric["persistent_reset_row_count"].append(
            float(telemetry.get("persistent_reset_row_count", 0.0))
        )
        numeric["persistent_delta_slot_count"].append(
            float(telemetry.get("persistent_delta_slot_count", 0.0))
        )
        numeric["persistent_delta_max_slots"].append(
            float(telemetry.get("persistent_delta_max_slots", 0.0))
        )
        numeric["persistent_update_fn_cache_size"].append(
            float(telemetry.get("persistent_update_fn_cache_size", 0.0))
        )
        numeric["persistent_partial_render_request"].append(
            float(telemetry.get("persistent_partial_render_request", 0.0))
        )
        numeric["lightzero_payload_pickle_bytes"].append(float(lightzero_payload_pickle_bytes))

    return {
        "schema_id": SCHEMA_ID,
        "impl_id": "curvyzero_modal_profile_only_renderer_backed_surface_facade/v0",
        "ok": True,
        "profile_only": True,
        "surface_facade_canary": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "app_name": APP_NAME,
        "renderer_app_name": RENDER_APP_NAME,
        "config": checked,
        "jax": {
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_count": len(jax.devices()),
        },
        "nvidia_smi": _nvidia_smi(),
        "packages": _package_versions(),
        "reset": {
            "surface_reset_total_sec": reset_total_sec,
            "policy_row_count": int(reset_step.info.get("policy_row_count", 0)),
            "renderer_telemetry": dict(
                reset_step.info.get("renderer_backed_stack_telemetry") or {}
            ),
        },
        "timings": _summarize_timings(per_step),
        "numeric": _summarize_numeric(numeric),
        "mock_collector": {
            "included": checked["include_lightzero_payload_profile"],
            "pickle_lightzero_payload": checked["pickle_lightzero_payload"],
            "include_rnd_meter": checked["include_rnd_meter"],
            "uses_surface_policy_rows": True,
            "rnd_metrics": None
            if rnd_model is None
            else rnd_model.metrics_snapshot(reason="surface_facade_profile"),
        },
        "divergence_canary": {
            "enabled": divergence_enabled,
            "reference_backend": None
            if not divergence_enabled
            else CpuOracleBatchedObservationRenderer.backend_name,
            "candidate_backend": getattr(surface.stack, "renderer_backend_name", None),
            "checked_reset": divergence_enabled,
            "checked_step_count": divergence_checks,
            "timings": _summarize_timings(divergence_timings) if divergence_enabled else {},
            **(
                _summarize_parity(divergence_summaries, config=checked)
                if divergence_enabled
                else {}
            ),
        },
        "known_gaps": [
            *_known_gaps(),
            "surface facade canary only; does not call stock train_muzero",
            "renderer-backed canary is profile-only; partial-row behavior must be checked separately before promotion",
        ],
    }


def _run_profile_env_manager_profile_impl(config: dict[str, Any]) -> dict[str, Any]:
    import jax
    import numpy as np

    if str(config.get("geometry_dtype", DEFAULT_BOUNDARY_GEOMETRY_DTYPE)) == "float64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    checked = _validate_boundary_config(np=np, config=config)
    render_mode_id = RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]
    bonus_render_mode_id = BONUS_RENDER_MODE_IDS[BONUS_RENDER_MODE_SIMPLE_SYMBOLS]
    render_fn_cache: dict[int, Any] = {}

    def render_fn_for_slots(render_trail_slots: int) -> Any:
        slots = int(render_trail_slots)
        cached = render_fn_cache.get(slots)
        if cached is not None:
            return cached
        slot_config = dict(checked["render_config"])
        slot_config["trail_slots"] = slots
        cached = _make_jax_two_view_render_fn(
            jax=jax,
            jnp=jnp,
            config=slot_config,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
        )
        render_fn_cache[slots] = cached
        return cached

    surface_stack_backend = str(checked["surface_stack_backend"])
    surface_kwargs: dict[str, Any] = {
        "batch_size": checked["batch_size"],
        "player_count": PLAYER_COUNT,
        "seed": checked["seed"],
        "decision_source_frames": 1,
        "source_physics_step_ms": SOURCE_PHYSICS_STEP_MS,
        "body_capacity": checked["body_capacity"],
        "map_size": checked["map_size"],
        "natural_bonus_spawn": False,
        "death_mode": checked["death_mode"],
        "max_ticks": checked["max_ticks"],
        "observation_stack_backend": surface_stack_backend,
    }
    if surface_stack_backend == TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE:
        renderer = _make_profile_observation_renderer(
            jax=jax,
            jnp=jnp,
            np=np,
            checked=checked,
            render_fn_for_slots=render_fn_for_slots,
            bonus_render_mode_id=bonus_render_mode_id,
        )
        surface_kwargs["observation_renderer"] = renderer
        surface_kwargs["required_observation_renderer_backend"] = renderer.backend_name
    surface = SourceStateMultiplayerTrainerSurface(**surface_kwargs)
    manager = BatchedLightZeroProfileEnvManager(BatchedLightZeroScalarActionBridge(surface))
    rng = np.random.default_rng(int(checked["seed"]))
    rnd_model = _build_rnd_model(checked) if checked["include_rnd_meter"] else None

    reset_started = time.perf_counter()
    manager.seed(int(checked["seed"]), dynamic_seed=False)
    manager.reset(int(checked["seed"]))
    reset_total_sec = time.perf_counter() - reset_started
    prewarm = {"enabled": False, "slots": [], "total_sec": 0.0}
    if (
        bool(checked["prewarm_render_functions"])
        and surface_stack_backend == TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE
    ):
        prewarm = _prewarm_dynamic_render_functions(
            jax=jax,
            np=np,
            production_state=surface.env.state,
            config=checked["render_config"],
            render_fn_for_slots=render_fn_for_slots,
        )

    per_step: dict[str, list[float]] = {
        "manager_step_total_sec": [],
        "surface_renderer_render_sec": [],
        "surface_renderer_device_render_sec": [],
        "surface_renderer_host_to_device_sec": [],
        "surface_renderer_device_to_host_sec": [],
        "surface_env_step_sec": [],
        "surface_stack_update_sec": [],
        "surface_reward_sec": [],
        "surface_package_sec": [],
        "manager_payload_pickle_sec": [],
        "rnd_collect_data_sec": [],
        "rnd_train_with_data_sec": [],
        "rnd_estimate_sec": [],
    }
    numeric: dict[str, list[float]] = {
        "ready_obs_count": [],
        "timestep_count": [],
        "manager_payload_pickle_bytes": [],
        "render_trail_slots": [],
        "active_trail_count_max": [],
        "render_truncation_row_count": [],
        "persistent_reset_row_count": [],
        "persistent_delta_slot_count": [],
        "persistent_delta_max_slots": [],
        "persistent_update_fn_cache_size": [],
        "persistent_partial_render_request": [],
        "terminal_timestep_count": [],
        "autoreset_row_count": [],
    }

    total_iterations = int(checked["warmup_steps"]) + int(checked["steps"])
    for iteration in range(total_iterations):
        ready_ids = tuple(sorted(int(env_id) for env_id in manager.ready_obs))
        action_by_env_id = {
            env_id: int(rng.integers(0, ACTION_COUNT, dtype=np.int16)) for env_id in ready_ids
        }
        started = time.perf_counter()
        result = manager.step(action_by_env_id)
        manager_step_total_sec = time.perf_counter() - started
        if iteration < int(checked["warmup_steps"]):
            continue

        telemetry = dict(
            result.bridge_output.surface_step.info.get("renderer_backed_stack_telemetry") or {}
        )
        surface_timing = dict(
            result.bridge_output.surface_step.info.get("trainer_surface_profile_timing") or {}
        )
        pickle_sec = 0.0
        pickle_bytes = 0
        if checked["pickle_lightzero_payload"]:
            started = time.perf_counter()
            payload = pickle.dumps(result.timestep_by_env_id, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_sec = time.perf_counter() - started
            pickle_bytes = len(payload)

        rnd_collect_data_sec = 0.0
        rnd_train_with_data_sec = 0.0
        rnd_estimate_sec = 0.0
        if rnd_model is not None and result.timestep_by_env_id:
            ordered_timesteps = [
                result.timestep_by_env_id[env_id] for env_id in sorted(result.timestep_by_env_id)
            ]
            flat_obs = np.ascontiguousarray(
                np.stack([item.obs["observation"] for item in ordered_timesteps], axis=0)
            )
            target_reward = np.asarray(
                [[float(item.reward)] for item in ordered_timesteps],
                dtype=np.float32,
            )
            segment = SimpleNamespace(obs_segment=flat_obs)
            started = time.perf_counter()
            rnd_model.collect_data([[segment]])
            rnd_collect_data_sec = time.perf_counter() - started

            started = time.perf_counter()
            rnd_model.train_with_data()
            rnd_train_with_data_sec = time.perf_counter() - started

            started = time.perf_counter()
            rnd_model.estimate([[flat_obs], [target_reward]])
            rnd_estimate_sec = time.perf_counter() - started

        per_step["manager_step_total_sec"].append(manager_step_total_sec)
        per_step["surface_renderer_render_sec"].append(float(telemetry.get("render_sec", 0.0)))
        per_step["surface_renderer_device_render_sec"].append(
            float(telemetry.get("device_render_sec", 0.0))
        )
        per_step["surface_renderer_host_to_device_sec"].append(
            float(telemetry.get("host_to_device_sec", 0.0))
        )
        per_step["surface_renderer_device_to_host_sec"].append(
            float(telemetry.get("device_to_host_sec", 0.0))
        )
        per_step["surface_env_step_sec"].append(float(surface_timing.get("env_step_sec", 0.0)))
        per_step["surface_stack_update_sec"].append(
            float(surface_timing.get("stack_update_sec", 0.0))
        )
        per_step["surface_reward_sec"].append(float(surface_timing.get("reward_sec", 0.0)))
        per_step["surface_package_sec"].append(float(surface_timing.get("package_sec", 0.0)))
        per_step["manager_payload_pickle_sec"].append(pickle_sec)
        per_step["rnd_collect_data_sec"].append(rnd_collect_data_sec)
        per_step["rnd_train_with_data_sec"].append(rnd_train_with_data_sec)
        per_step["rnd_estimate_sec"].append(rnd_estimate_sec)
        numeric["ready_obs_count"].append(float(len(result.ready_obs)))
        numeric["timestep_count"].append(float(len(result.timestep_by_env_id)))
        numeric["manager_payload_pickle_bytes"].append(float(pickle_bytes))
        numeric["render_trail_slots"].append(float(telemetry.get("render_trail_slots", 0.0)))
        numeric["active_trail_count_max"].append(
            float(telemetry.get("active_trail_count_max", 0.0))
        )
        numeric["render_truncation_row_count"].append(
            float(telemetry.get("render_truncation_row_count", 0.0))
        )
        numeric["persistent_reset_row_count"].append(
            float(telemetry.get("persistent_reset_row_count", 0.0))
        )
        numeric["persistent_delta_slot_count"].append(
            float(telemetry.get("persistent_delta_slot_count", 0.0))
        )
        numeric["persistent_delta_max_slots"].append(
            float(telemetry.get("persistent_delta_max_slots", 0.0))
        )
        numeric["persistent_update_fn_cache_size"].append(
            float(telemetry.get("persistent_update_fn_cache_size", 0.0))
        )
        numeric["persistent_partial_render_request"].append(
            float(telemetry.get("persistent_partial_render_request", 0.0))
        )
        numeric["terminal_timestep_count"].append(
            float(
                sum(
                    1
                    for item in result.timestep_by_env_id.values()
                    if bool(np.asarray(item.done).item())
                )
            )
        )
        numeric["autoreset_row_count"].append(
            float(np.count_nonzero(result.bridge_output.autoreset_row_mask))
        )

    return {
        "schema_id": SCHEMA_ID,
        "impl_id": "curvyzero_modal_profile_only_batched_lightzero_env_manager/v0",
        "ok": True,
        "profile_only": True,
        "profile_env_manager_canary": True,
        "surface_facade_canary": False,
        "calls_train_muzero": False,
        "stock_lightzero_integrated": False,
        "touches_live_runs": False,
        "app_name": APP_NAME,
        "renderer_app_name": RENDER_APP_NAME,
        "config": checked,
        "manager": {
            "env_num": int(manager.env_num),
            "ready_obs_count": len(manager.ready_obs),
            "last_reset_info_count": len(manager.last_reset_info),
            "scalar_env_instances_created": 0,
            "vector_surface_batch_size": int(surface.batch_size),
            "player_count": int(surface.player_count),
            "renderer_backend": getattr(surface.stack, "renderer_backend_name", None),
            "trainer_observation_no_hidden_fallback": True,
        },
        "jax": {
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_count": len(jax.devices()),
        },
        "nvidia_smi": _nvidia_smi(),
        "packages": _package_versions(),
        "reset": {"manager_reset_total_sec": reset_total_sec},
        "prewarm": prewarm,
        "timings": _summarize_timings(per_step),
        "numeric": _summarize_numeric(numeric),
        "mock_collector": {
            "included": True,
            "profile_env_manager_shape": True,
            "pickle_lightzero_payload": checked["pickle_lightzero_payload"],
            "include_rnd_meter": checked["include_rnd_meter"],
            "rnd_metrics": None
            if rnd_model is None
            else rnd_model.metrics_snapshot(reason="profile_env_manager"),
        },
        "known_gaps": [
            *_known_gaps(),
            "manager facade canary only; does not call stock train_muzero",
            "base-manager shape is local/profile-only until wired into LightZero",
        ],
    }


def _run_hybrid_observation_profile_impl(config: dict[str, Any]) -> dict[str, Any]:
    import jax
    import numpy as np

    if str(config.get("geometry_dtype", DEFAULT_BOUNDARY_GEOMETRY_DTYPE)) == "float64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    checked = _validate_boundary_config(np=np, config=config)
    if bool(checked["include_rnd_meter"]):
        raise ValueError(
            "hybrid_observation_canary does not include RND; run RND as a separate axis"
        )
    actor_count = int(checked["actor_count"])
    policy_probe_simulations = _nonnegative_int(
        config.get("hybrid_policy_probe_simulations", 0),
        "hybrid_policy_probe_simulations",
    )
    policy_probe_channels = _positive_int(
        config.get("hybrid_policy_probe_channels", 16),
        "hybrid_policy_probe_channels",
    )
    batched_stack_probe_simulations = _nonnegative_int(
        config.get("hybrid_batched_stack_probe_simulations", 0),
        "hybrid_batched_stack_probe_simulations",
    )
    batched_stack_probe_channels = _positive_int(
        config.get("hybrid_batched_stack_probe_channels", 16),
        "hybrid_batched_stack_probe_channels",
    )
    materialize_scalar_timestep = bool(config.get("hybrid_materialize_scalar_timestep", True))
    stack_storage_dtype = str(config.get("hybrid_stack_storage_dtype", "float32"))
    if stack_storage_dtype not in HYBRID_STACK_STORAGE_DTYPES:
        allowed = ", ".join(HYBRID_STACK_STORAGE_DTYPES)
        raise ValueError(
            f"hybrid_stack_storage_dtype must be one of {allowed}; got {stack_storage_dtype!r}"
        )
    compact_service_replay_proof = bool(config.get("hybrid_compact_service_replay_proof", False))
    compact_rollout_slab_probe = bool(config.get("hybrid_compact_rollout_slab_probe", False))
    compact_rollout_slab_sample_gate = bool(
        config.get("hybrid_compact_rollout_slab_sample_gate", False)
    )
    compact_rollout_slab_sample_gate_batch_size = _nonnegative_int(
        config.get("hybrid_compact_rollout_slab_sample_gate_batch_size", 0),
        "hybrid_compact_rollout_slab_sample_gate_batch_size",
    )
    compact_rollout_slab_sample_gate_interval = _positive_int(
        config.get("hybrid_compact_rollout_slab_sample_gate_interval", 1),
        "hybrid_compact_rollout_slab_sample_gate_interval",
    )
    compact_rollout_slab_sample_gate_replay_pair_capacity = _positive_int(
        config["hybrid_compact_rollout_slab_sample_gate_replay_pair_capacity"],
        "hybrid_compact_rollout_slab_sample_gate_replay_pair_capacity",
    )
    compact_rollout_slab_learner_gate = bool(
        config.get("hybrid_compact_rollout_slab_learner_gate", False)
    )
    compact_rollout_slab_learner_gate_train_steps = _positive_int(
        config.get("hybrid_compact_rollout_slab_learner_gate_train_steps", 1),
        "hybrid_compact_rollout_slab_learner_gate_train_steps",
    )
    compact_rollout_slab_learner_gate_device = str(
        config.get("hybrid_compact_rollout_slab_learner_gate_device", "cuda")
    )
    if compact_rollout_slab_learner_gate_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "hybrid_compact_rollout_slab_learner_gate_device must be 'auto', 'cpu', or 'cuda'"
        )
    compact_rollout_slab_learner_gate_include_rnd = bool(
        config.get("hybrid_compact_rollout_slab_learner_gate_include_rnd", False)
    )
    compact_rollout_slab_learner_gate_impl = str(
        config.get(
            "hybrid_compact_rollout_slab_learner_gate_impl",
            COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_TOY_PROBE,
        )
    )
    if compact_rollout_slab_learner_gate_impl not in COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPLS:
        allowed = ", ".join(COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPLS)
        raise ValueError(
            "hybrid_compact_rollout_slab_learner_gate_impl must be one of "
            f"{allowed}; got {compact_rollout_slab_learner_gate_impl!r}"
        )
    compact_rollout_slab_learner_gate_support_scale = _positive_int(
        config.get("hybrid_compact_rollout_slab_learner_gate_support_scale", 1),
        "hybrid_compact_rollout_slab_learner_gate_support_scale",
    )
    compact_rollout_slab_learner_gate_num_unroll_steps = _positive_int(
        config.get("hybrid_compact_rollout_slab_learner_gate_num_unroll_steps", 1),
        "hybrid_compact_rollout_slab_learner_gate_num_unroll_steps",
    )
    if (
        compact_rollout_slab_learner_gate_num_unroll_steps != 1
        and compact_rollout_slab_learner_gate_impl
        != COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO
    ):
        raise ValueError(
            "hybrid_compact_rollout_slab_learner_gate_num_unroll_steps > 1 "
            "requires compact_muzero learner gate"
        )
    compact_owned_loop_entrypoint = bool(config.get("hybrid_compact_owned_loop_entrypoint", False))
    compact_owned_loop_policy_version_ref = str(
        config.get("hybrid_compact_owned_loop_policy_version_ref", "")
    ).strip()
    compact_owned_loop_model_version_ref_raw = str(
        config.get("hybrid_compact_owned_loop_model_version_ref", "")
    ).strip()
    compact_owned_loop_model_version_ref = compact_owned_loop_model_version_ref_raw or None
    compact_owned_loop_policy_source = str(
        config.get("hybrid_compact_owned_loop_policy_source", "")
    ).strip()
    compact_owned_loop_capture_replay_store_state = bool(
        config.get("hybrid_compact_owned_loop_capture_replay_store_state", False)
    )
    compact_rollout_slab_action_mode = str(
        config.get(
            "hybrid_compact_rollout_slab_action_mode",
            COMPACT_ROLLOUT_SLAB_ACTION_MODE_SEARCH_FEEDBACK,
        )
    )
    if compact_rollout_slab_action_mode not in COMPACT_ROLLOUT_SLAB_ACTION_MODES:
        allowed = ", ".join(COMPACT_ROLLOUT_SLAB_ACTION_MODES)
        raise ValueError(
            "hybrid_compact_rollout_slab_action_mode must be one of "
            f"{allowed}; got {compact_rollout_slab_action_mode!r}"
        )
    compact_root_tape_compare = bool(checked.get("hybrid_compact_root_tape_compare", False))
    compact_root_tape_max_records = _positive_int(
        checked.get("hybrid_compact_root_tape_max_records", 4),
        "hybrid_compact_root_tape_max_records",
    )
    compact_root_tape_allow_resident_host_snapshot = bool(
        checked.get("hybrid_compact_root_tape_allow_resident_host_snapshot", False)
    )
    compact_root_tape_compare_fixed_shape_floor = bool(
        checked.get("hybrid_compact_root_tape_compare_fixed_shape_floor", True)
    )
    compact_root_tape_compare_mctx = bool(
        checked.get("hybrid_compact_root_tape_compare_mctx", False)
    )
    compact_root_tape_compare_model_compile = bool(
        checked.get("hybrid_compact_root_tape_compare_model_compile", False)
    )
    compact_root_tape_compare_direct_core = bool(
        checked.get("hybrid_compact_root_tape_compare_direct_core", False)
    )
    compact_root_tape_model_compile_mode = str(
        checked.get("hybrid_compact_root_tape_model_compile_mode", "default")
    )
    compact_root_tape_require_model_compile = bool(
        checked.get("hybrid_compact_root_tape_require_model_compile", True)
    )
    compact_root_tape_reference_label = str(
        checked.get("hybrid_compact_root_tape_reference_label", "primary")
    )
    if compact_rollout_slab_probe and compact_service_replay_proof:
        raise ValueError(
            "hybrid_compact_rollout_slab_probe owns selected-action feedback and "
            "replay index commits; do not also enable hybrid_compact_service_replay_proof"
        )
    if compact_rollout_slab_sample_gate and not compact_rollout_slab_probe:
        raise ValueError(
            "hybrid_compact_rollout_slab_sample_gate requires hybrid_compact_rollout_slab_probe"
        )
    if compact_rollout_slab_learner_gate and not compact_rollout_slab_sample_gate:
        raise ValueError(
            "hybrid_compact_rollout_slab_learner_gate requires "
            "hybrid_compact_rollout_slab_sample_gate"
        )
    if compact_owned_loop_entrypoint:
        if not compact_rollout_slab_probe:
            raise ValueError(
                "hybrid_compact_owned_loop_entrypoint requires hybrid_compact_rollout_slab_probe"
            )
        if not compact_rollout_slab_sample_gate:
            raise ValueError(
                "hybrid_compact_owned_loop_entrypoint requires "
                "hybrid_compact_rollout_slab_sample_gate"
            )
        if not compact_rollout_slab_learner_gate:
            raise ValueError(
                "hybrid_compact_owned_loop_entrypoint requires "
                "hybrid_compact_rollout_slab_learner_gate"
            )
        if materialize_scalar_timestep:
            raise ValueError(
                "hybrid_compact_owned_loop_entrypoint requires "
                "hybrid_materialize_scalar_timestep=false"
            )
        if not compact_owned_loop_policy_version_ref:
            raise ValueError("hybrid_compact_owned_loop_policy_version_ref must be non-empty")
        if not compact_owned_loop_policy_source:
            raise ValueError("hybrid_compact_owned_loop_policy_source must be non-empty")
    hybrid_device_only_stack = bool(checked.get("hybrid_device_only_stack", False))
    hybrid_refresh_observation_stack = bool(checked.get("hybrid_refresh_observation_stack", True))
    hybrid_resident_observation_search = bool(
        checked.get("hybrid_resident_observation_search", False)
    )
    if hybrid_resident_observation_search:
        if not hybrid_device_only_stack:
            raise ValueError("hybrid_resident_observation_search requires hybrid_device_only_stack")
        if materialize_scalar_timestep:
            raise ValueError(
                "hybrid_resident_observation_search requires "
                "hybrid_materialize_scalar_timestep=False"
            )
        if stack_storage_dtype != "uint8":
            raise ValueError(
                "hybrid_resident_observation_search requires hybrid_stack_storage_dtype='uint8'"
            )
        if (
            str(checked["observation_renderer_backend"])
            != SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
        ):
            raise ValueError(
                "hybrid_resident_observation_search requires "
                f"{SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND!r}"
            )
        if (
            str(checked.get("hybrid_lightzero_array_ceiling_mode", ""))
            != LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
        ):
            raise ValueError(
                "hybrid_resident_observation_search currently requires "
                "hybrid_lightzero_array_ceiling_mode='compact_torch_search_service'"
            )
    hybrid_native_actor_buffer = bool(checked.get("hybrid_native_actor_buffer", False))
    persistent_compact_render_state_buffer = bool(
        checked.get("hybrid_persistent_compact_render_state_buffer", False)
    )
    borrow_single_actor_render_state = bool(
        checked.get("hybrid_borrow_single_actor_render_state", False)
    )
    if persistent_compact_render_state_buffer and (
        str(checked["observation_renderer_backend"])
        != SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
    ):
        raise ValueError(
            "hybrid_persistent_compact_render_state_buffer requires "
            f"{SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND!r}"
        )
    device_latest_stack_probe = bool(config.get("hybrid_batched_stack_probe_device_latest", False))
    if device_latest_stack_probe and (
        str(checked["observation_renderer_backend"])
        != SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
    ):
        raise ValueError(
            "hybrid_batched_stack_probe_device_latest requires "
            f"{SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND!r}"
        )
    resident_chunk_probe = bool(config.get("hybrid_resident_chunk_probe", False))
    resident_replay_steps = _positive_int(
        config.get("hybrid_resident_replay_steps", 64),
        "hybrid_resident_replay_steps",
    )
    resident_sample_batch_size = _positive_int(
        config.get("hybrid_resident_sample_batch_size", 256),
        "hybrid_resident_sample_batch_size",
    )
    resident_replay_train_steps = _positive_int(
        config.get("hybrid_resident_replay_train_steps", 1),
        "hybrid_resident_replay_train_steps",
    )
    resident_readback_checksum = bool(config.get("hybrid_resident_readback_checksum", True))
    lightzero_collect_forward_probe = bool(
        config.get("hybrid_lightzero_collect_forward_probe", False)
    )
    lightzero_initial_inference_probe = bool(
        config.get("hybrid_lightzero_initial_inference_probe", False)
    )
    lightzero_array_ceiling_probe = bool(config.get("hybrid_lightzero_array_ceiling_probe", False))
    lightzero_mcts_arrays_boundary_probe = bool(
        config.get("hybrid_lightzero_mcts_arrays_boundary_probe", False)
    )
    mctx_compact_search_probe = bool(config.get("hybrid_mctx_compact_search_probe", False))
    lightzero_mcts_arrays_boundary_impl = str(
        config.get(
            "hybrid_lightzero_mcts_arrays_boundary_impl",
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_STOCK_FACADE,
        )
    )
    if lightzero_mcts_arrays_boundary_impl not in LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPLS:
        allowed = ", ".join(LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPLS)
        raise ValueError(
            "hybrid_lightzero_mcts_arrays_boundary_impl must be one of "
            f"{allowed}; got {lightzero_mcts_arrays_boundary_impl!r}"
        )
    lightzero_mcts_arrays_boundary_input_mode = str(
        config.get(
            "hybrid_lightzero_mcts_arrays_boundary_input_mode",
            LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8,
        )
    )
    if lightzero_mcts_arrays_boundary_input_mode not in LIGHTZERO_ARRAY_CEILING_INPUT_MODES:
        allowed = ", ".join(LIGHTZERO_ARRAY_CEILING_INPUT_MODES)
        raise ValueError(
            "hybrid_lightzero_mcts_arrays_boundary_input_mode must be one of "
            f"{allowed}; got {lightzero_mcts_arrays_boundary_input_mode!r}"
        )
    lightzero_array_ceiling_mode = str(
        config.get(
            "hybrid_lightzero_array_ceiling_mode",
            LIGHTZERO_ARRAY_CEILING_MODE_POLICY_ARRAYS,
        )
    )
    if lightzero_array_ceiling_mode not in LIGHTZERO_ARRAY_CEILING_MODES:
        allowed = ", ".join(LIGHTZERO_ARRAY_CEILING_MODES)
        raise ValueError(
            "hybrid_lightzero_array_ceiling_mode must be one of "
            f"{allowed}; got {lightzero_array_ceiling_mode!r}"
        )
    lightzero_array_ceiling_input_mode = str(
        config.get(
            "hybrid_lightzero_array_ceiling_input_mode",
            LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8,
        )
    )
    if lightzero_array_ceiling_input_mode not in LIGHTZERO_ARRAY_CEILING_INPUT_MODES:
        allowed = ", ".join(LIGHTZERO_ARRAY_CEILING_INPUT_MODES)
        raise ValueError(
            "hybrid_lightzero_array_ceiling_input_mode must be one of "
            f"{allowed}; got {lightzero_array_ceiling_input_mode!r}"
        )
    lightzero_mock_service_materialize_public_output = bool(
        config.get("hybrid_lightzero_mock_service_materialize_public_output", False)
    )
    if (
        lightzero_mock_service_materialize_public_output
        and lightzero_array_ceiling_mode != LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE
    ):
        raise ValueError(
            "hybrid_lightzero_mock_service_materialize_public_output requires "
            "hybrid_lightzero_array_ceiling_mode='mock_search_service'"
        )
    lightzero_consumer_num_simulations = _positive_int(
        config.get(
            "hybrid_lightzero_consumer_num_simulations", max(1, batched_stack_probe_simulations)
        ),
        "hybrid_lightzero_consumer_num_simulations",
    )
    lightzero_consumer_temperature = float(config.get("hybrid_lightzero_consumer_temperature", 1.0))
    lightzero_consumer_epsilon = float(config.get("hybrid_lightzero_consumer_epsilon", 0.0))
    lightzero_consumer_root_noise_weight_raw = float(
        config.get("hybrid_lightzero_consumer_root_noise_weight", -1.0)
    )
    lightzero_consumer_root_noise_weight = (
        None
        if lightzero_consumer_root_noise_weight_raw < 0.0
        else lightzero_consumer_root_noise_weight_raw
    )
    lightzero_consumer_use_cuda = bool(config.get("hybrid_lightzero_consumer_use_cuda", True))
    lightzero_consumer_collect_with_pure_policy = bool(
        config.get("hybrid_lightzero_consumer_collect_with_pure_policy", False)
    )
    compact_torch_compile_search = bool(config.get("hybrid_compact_torch_compile_search", True))
    compact_torch_compile_model_inference = bool(
        config.get("hybrid_compact_torch_compile_model_inference", False)
    )
    compact_torch_require_model_compile = bool(
        config.get("hybrid_compact_torch_require_model_compile", False)
    )
    compact_torch_model_compile_mode = str(
        config.get("hybrid_compact_torch_model_compile_mode", "reduce-overhead")
    )
    if compact_torch_model_compile_mode not in COMPACT_TORCH_MODEL_COMPILE_MODES:
        allowed = ", ".join(COMPACT_TORCH_MODEL_COMPILE_MODES)
        raise ValueError(
            "hybrid_compact_torch_model_compile_mode must be one of "
            f"{allowed}; got {compact_torch_model_compile_mode!r}"
        )
    compact_torch_recurrent_action_shape_mode = str(
        config.get("hybrid_compact_torch_recurrent_action_shape_mode", "auto")
    )
    if compact_torch_recurrent_action_shape_mode not in {"auto", "flat", "column"}:
        raise ValueError(
            "hybrid_compact_torch_recurrent_action_shape_mode must be auto, flat, "
            f"or column; got {compact_torch_recurrent_action_shape_mode!r}"
        )
    compact_torch_timing_mode = str(
        config.get("hybrid_compact_torch_timing_mode", "host_phase_sync")
    )
    if compact_torch_timing_mode not in COMPACT_TORCH_TIMING_MODES:
        allowed = ", ".join(COMPACT_TORCH_TIMING_MODES)
        raise ValueError(
            "hybrid_compact_torch_timing_mode must be one of "
            f"{allowed}; got {compact_torch_timing_mode!r}"
        )
    compact_torch_initial_inference_mode = str(
        config.get("hybrid_compact_torch_initial_inference_mode", "model_method")
    )
    if compact_torch_initial_inference_mode not in COMPACT_TORCH_INITIAL_INFERENCE_MODES:
        allowed = ", ".join(COMPACT_TORCH_INITIAL_INFERENCE_MODES)
        raise ValueError(
            "hybrid_compact_torch_initial_inference_mode must be one of "
            f"{allowed}; got {compact_torch_initial_inference_mode!r}"
        )
    compact_torch_observation_memory_format = str(
        config.get("hybrid_compact_torch_observation_memory_format", "contiguous")
    )
    if compact_torch_observation_memory_format not in COMPACT_TORCH_MEMORY_FORMATS:
        allowed = ", ".join(COMPACT_TORCH_MEMORY_FORMATS)
        raise ValueError(
            "hybrid_compact_torch_observation_memory_format must be one of "
            f"{allowed}; got {compact_torch_observation_memory_format!r}"
        )
    compact_torch_model_memory_format = str(
        config.get("hybrid_compact_torch_model_memory_format", "contiguous")
    )
    if compact_torch_model_memory_format not in COMPACT_TORCH_MEMORY_FORMATS:
        allowed = ", ".join(COMPACT_TORCH_MEMORY_FORMATS)
        raise ValueError(
            "hybrid_compact_torch_model_memory_format must be one of "
            f"{allowed}; got {compact_torch_model_memory_format!r}"
        )
    if compact_torch_model_memory_format != "contiguous":
        raise ValueError(
            "hybrid_compact_torch_model_memory_format=channels_last is parked for the "
            "current LightZero MuZero model because recurrent dynamics uses .view(); "
            "use hybrid_compact_torch_model_memory_format='contiguous'"
        )
    if hybrid_resident_observation_search:
        if not hybrid_device_only_stack:
            raise ValueError(
                "hybrid_resident_observation_search requires hybrid_device_only_stack=True"
            )
        if stack_storage_dtype != "uint8":
            raise ValueError(
                "hybrid_resident_observation_search requires hybrid_stack_storage_dtype='uint8'"
            )
        if materialize_scalar_timestep:
            raise ValueError(
                "hybrid_resident_observation_search requires "
                "hybrid_materialize_scalar_timestep=False"
            )
        if not lightzero_consumer_use_cuda:
            raise ValueError(
                "hybrid_resident_observation_search requires "
                "hybrid_lightzero_consumer_use_cuda=True"
            )
        if not lightzero_array_ceiling_probe:
            raise ValueError(
                "hybrid_resident_observation_search currently requires "
                "hybrid_lightzero_array_ceiling_probe"
            )
        if (
            lightzero_array_ceiling_mode
            != LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
        ):
            raise ValueError(
                "hybrid_resident_observation_search currently requires "
                "hybrid_lightzero_array_ceiling_mode="
                f"{LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE!r}"
            )
    mctx_num_simulations = _positive_int(
        config.get("hybrid_mctx_num_simulations", lightzero_consumer_num_simulations),
        "hybrid_mctx_num_simulations",
    )
    mctx_hidden_dim = _positive_int(
        config.get("hybrid_mctx_hidden_dim", 64),
        "hybrid_mctx_hidden_dim",
    )
    mctx_visual_channels = _positive_int(
        config.get("hybrid_mctx_visual_channels", 8),
        "hybrid_mctx_visual_channels",
    )
    mctx_require_gpu_backend = bool(config.get("hybrid_mctx_require_gpu_backend", True))
    mctx_lightzero_checkpoint_ref = str(checked.get("hybrid_mctx_lightzero_checkpoint_ref", ""))
    mctx_lightzero_checkpoint_state_key = str(
        checked.get("hybrid_mctx_lightzero_checkpoint_state_key", "")
    )
    mctx_compare_direct_ctree = bool(checked.get("hybrid_mctx_compare_direct_ctree", False))
    mctx_compare_direct_ctree_impl = str(
        checked.get(
            "hybrid_mctx_compare_direct_ctree_impl",
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
        )
    )
    pre_scalar_consumer_count = sum(
        [
            bool(resident_chunk_probe),
            bool(lightzero_collect_forward_probe),
            bool(lightzero_initial_inference_probe),
            bool(lightzero_array_ceiling_probe),
            bool(lightzero_mcts_arrays_boundary_probe),
            bool(mctx_compact_search_probe),
            bool(
                batched_stack_probe_simulations > 0
                and not lightzero_collect_forward_probe
                and not lightzero_initial_inference_probe
                and not lightzero_array_ceiling_probe
                and not lightzero_mcts_arrays_boundary_probe
                and not mctx_compact_search_probe
            ),
        ]
    )
    if pre_scalar_consumer_count > 1:
        raise ValueError(
            "choose one pre-scalar hybrid consumer: synthetic batched-stack, "
            "resident chunk, LightZero collect-forward, LightZero initial-inference, "
            "LightZero array-ceiling, LightZero MCTS arrays-boundary, or MCTX compact search"
        )
    if (
        lightzero_collect_forward_probe
        or lightzero_initial_inference_probe
        or lightzero_array_ceiling_probe
        or lightzero_mcts_arrays_boundary_probe
        or mctx_compact_search_probe
    ):
        probe_name = (
            "hybrid_lightzero_collect_forward_probe"
            if lightzero_collect_forward_probe
            else (
                "hybrid_lightzero_initial_inference_probe"
                if lightzero_initial_inference_probe
                else (
                    "hybrid_lightzero_array_ceiling_probe"
                    if lightzero_array_ceiling_probe
                    else (
                        "hybrid_lightzero_mcts_arrays_boundary_probe"
                        if lightzero_mcts_arrays_boundary_probe
                        else "hybrid_mctx_compact_search_probe"
                    )
                )
            )
        )
        if device_latest_stack_probe:
            raise ValueError(
                f"{probe_name} starts from the profile host stack; "
                "run device-latest as a separate follow-up axis"
            )
        if stack_storage_dtype != "uint8":
            raise ValueError(f"{probe_name} requires hybrid_stack_storage_dtype='uint8'")
    if lightzero_collect_forward_probe or lightzero_mcts_arrays_boundary_probe:
        if lightzero_consumer_temperature <= 0.0:
            raise ValueError("hybrid_lightzero_consumer_temperature must be positive")
        if not 0.0 <= lightzero_consumer_epsilon <= 1.0:
            raise ValueError("hybrid_lightzero_consumer_epsilon must be in [0, 1]")
        if (
            lightzero_consumer_root_noise_weight is not None
            and not 0.0 <= lightzero_consumer_root_noise_weight <= 1.0
        ):
            raise ValueError("hybrid_lightzero_consumer_root_noise_weight must be -1 or in [0, 1]")
    compact_search_from_direct = bool(lightzero_mcts_arrays_boundary_probe)
    compact_search_from_array_ceiling = bool(
        lightzero_array_ceiling_probe
        and lightzero_array_ceiling_mode
        in {
            LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE,
            LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE,
            LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE,
            LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER,
            *LIGHTZERO_ARRAY_CEILING_DENSE_TORCH_MCTS_MODES,
        }
    )
    compact_search_from_mctx = bool(mctx_compact_search_probe)
    if mctx_compact_search_probe and not compact_rollout_slab_probe:
        raise ValueError(
            "hybrid_mctx_compact_search_probe currently runs only behind "
            "hybrid_compact_rollout_slab_probe"
        )
    if compact_service_replay_proof and mctx_compact_search_probe:
        raise ValueError(
            "hybrid_mctx_compact_search_probe does not support "
            "hybrid_compact_service_replay_proof; use compact rollout slab"
        )
    if compact_service_replay_proof or compact_rollout_slab_probe:
        if not (
            compact_search_from_direct
            or compact_search_from_array_ceiling
            or compact_search_from_mctx
        ):
            raise ValueError(
                "hybrid compact search proof requires "
                "direct CTree arrays, compact array-ceiling search probe, "
                "or MCTX compact search probe"
            )
        replay_input_mode = (
            LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8
            if compact_search_from_mctx
            else (
                lightzero_array_ceiling_input_mode
                if compact_search_from_array_ceiling
                else lightzero_mcts_arrays_boundary_input_mode
            )
        )
        if replay_input_mode == LIGHTZERO_ARRAY_CEILING_INPUT_MODE_RESIDENT_TORCH_REUSE:
            raise ValueError(
                "hybrid compact search proof requires fresh inputs; "
                "resident_torch_reuse is a stale-input ceiling"
            )
        if compact_search_from_direct:
            if lightzero_mcts_arrays_boundary_impl not in {
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT,
            }:
                raise ValueError("hybrid compact search proof requires a direct CTree arrays impl")
    if compact_rollout_slab_sample_gate and materialize_scalar_timestep:
        raise ValueError(
            "hybrid_compact_rollout_slab_sample_gate is a no-scalar collection proof; "
            "set --no-hybrid-materialize-scalar-timestep"
        )
    if resident_chunk_probe:
        if device_latest_stack_probe:
            raise ValueError(
                "hybrid_resident_chunk_probe starts from host-fed uint8 stacks; "
                "run device-latest as a separate follow-up axis"
            )
        if stack_storage_dtype != "uint8":
            raise ValueError(
                "hybrid_resident_chunk_probe requires hybrid_stack_storage_dtype='uint8'"
            )

    render_mode_id = RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]
    bonus_render_mode_id = BONUS_RENDER_MODE_IDS[BONUS_RENDER_MODE_SIMPLE_SYMBOLS]
    render_fn_cache: dict[int, Any] = {}

    def render_fn_for_slots(render_trail_slots: int) -> Any:
        slots = int(render_trail_slots)
        cached = render_fn_cache.get(slots)
        if cached is not None:
            return cached
        slot_config = dict(checked["render_config"])
        slot_config["trail_slots"] = slots
        cached = _make_jax_two_view_render_fn(
            jax=jax,
            jnp=jnp,
            config=slot_config,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
        )
        render_fn_cache[slots] = cached
        return cached

    renderer = _make_profile_observation_renderer(
        jax=jax,
        jnp=jnp,
        np=np,
        checked=checked,
        render_fn_for_slots=render_fn_for_slots,
        bonus_render_mode_id=bonus_render_mode_id,
    )
    policy_search_probe = None
    if policy_probe_simulations > 0:
        policy_search_probe = _JaxHybridPolicySearchProbe(
            jax=jax,
            jnp=jnp,
            simulations=policy_probe_simulations,
            channels=policy_probe_channels,
        )
    batched_stack_probe = None
    if resident_chunk_probe:
        batched_stack_probe = _JaxHybridResidentChunkProbe(
            jax=jax,
            jnp=jnp,
            simulations=max(1, batched_stack_probe_simulations),
            channels=batched_stack_probe_channels,
            replay_steps=resident_replay_steps,
            sample_batch_size=resident_sample_batch_size,
            replay_train_steps=resident_replay_train_steps,
            readback_checksum=resident_readback_checksum,
        )
    elif lightzero_collect_forward_probe:
        policy_info = _build_profile_lightzero_policy(
            seed=int(checked["seed"]),
            use_cuda=lightzero_consumer_use_cuda,
            num_simulations=lightzero_consumer_num_simulations,
            collect_with_pure_policy=lightzero_consumer_collect_with_pure_policy,
            policy_batch_size=int(checked["batch_size"]) * PLAYER_COUNT,
            max_ticks=int(checked["max_ticks"]),
            root_noise_weight=lightzero_consumer_root_noise_weight,
        )
        batched_stack_probe = _LightZeroCollectForwardStackProbe(
            policy=policy_info["policy"],
            policy_metadata=policy_info,
            num_simulations=lightzero_consumer_num_simulations,
            temperature=lightzero_consumer_temperature,
            epsilon=lightzero_consumer_epsilon,
        )
    elif lightzero_mcts_arrays_boundary_probe:
        policy_info = _build_profile_lightzero_policy(
            seed=int(checked["seed"]),
            use_cuda=lightzero_consumer_use_cuda,
            num_simulations=lightzero_consumer_num_simulations,
            collect_with_pure_policy=False,
            policy_batch_size=int(checked["batch_size"]) * PLAYER_COUNT,
            max_ticks=int(checked["max_ticks"]),
            root_noise_weight=lightzero_consumer_root_noise_weight,
        )
        batched_stack_probe = _LightZeroCollectForwardStackProbe(
            policy=policy_info["policy"],
            policy_metadata=policy_info,
            num_simulations=lightzero_consumer_num_simulations,
            temperature=lightzero_consumer_temperature,
            epsilon=lightzero_consumer_epsilon,
            arrays_boundary=True,
            arrays_boundary_impl=lightzero_mcts_arrays_boundary_impl,
            input_mode=lightzero_mcts_arrays_boundary_input_mode,
        )
    elif lightzero_initial_inference_probe:
        policy_info = _build_profile_lightzero_policy(
            seed=int(checked["seed"]),
            use_cuda=lightzero_consumer_use_cuda,
            num_simulations=lightzero_consumer_num_simulations,
            collect_with_pure_policy=lightzero_consumer_collect_with_pure_policy,
            policy_batch_size=int(checked["batch_size"]) * PLAYER_COUNT,
            max_ticks=int(checked["max_ticks"]),
            root_noise_weight=lightzero_consumer_root_noise_weight,
        )
        batched_stack_probe = _LightZeroInitialInferenceStackProbe(
            policy=policy_info["policy"],
            policy_metadata=policy_info,
        )
    elif lightzero_array_ceiling_probe:
        policy_info = _build_profile_lightzero_policy(
            seed=int(checked["seed"]),
            use_cuda=lightzero_consumer_use_cuda,
            num_simulations=lightzero_consumer_num_simulations,
            collect_with_pure_policy=False,
            policy_batch_size=int(checked["batch_size"]) * PLAYER_COUNT,
            max_ticks=int(checked["max_ticks"]),
            root_noise_weight=lightzero_consumer_root_noise_weight,
        )
        batched_stack_probe = _LightZeroArrayCeilingStackProbe(
            policy=policy_info["policy"],
            policy_metadata=policy_info,
            num_simulations=lightzero_consumer_num_simulations,
            mode=lightzero_array_ceiling_mode,
            input_mode=lightzero_array_ceiling_input_mode,
            materialize_public_output=lightzero_mock_service_materialize_public_output,
            require_resident_observation=hybrid_resident_observation_search,
            compile_search=compact_torch_compile_search,
            compile_model_inference=compact_torch_compile_model_inference,
            require_model_compile=compact_torch_require_model_compile,
            model_compile_mode=compact_torch_model_compile_mode,
            recurrent_action_shape_mode=compact_torch_recurrent_action_shape_mode,
            timing_mode=compact_torch_timing_mode,
            initial_inference_mode=compact_torch_initial_inference_mode,
            observation_memory_format=compact_torch_observation_memory_format,
            model_memory_format=compact_torch_model_memory_format,
        )
    elif batched_stack_probe_simulations > 0:
        batched_stack_probe = _JaxHybridBatchedStackProbe(
            jax=jax,
            jnp=jnp,
            simulations=batched_stack_probe_simulations,
            channels=batched_stack_probe_channels,
            device_latest_provider=renderer if device_latest_stack_probe else None,
        )
    compact_rollout_slab = None
    compact_root_tape_recorder = None
    compact_root_tape_services: dict[str, Any] = {}
    manager_batched_stack_probe = batched_stack_probe
    if compact_rollout_slab_probe:
        if batched_stack_probe is None and not mctx_compact_search_probe:
            raise ValueError("hybrid_compact_rollout_slab_probe requires a compact search probe")
        if lightzero_mcts_arrays_boundary_probe:
            search_service = _LightZeroCollectForwardCompactSearchService(batched_stack_probe)
            search_lane = str(lightzero_mcts_arrays_boundary_impl)
        elif lightzero_array_ceiling_probe:
            search_service = _LightZeroArrayCeilingCompactSearchService(batched_stack_probe)
            search_lane = str(lightzero_array_ceiling_mode)
        elif mctx_compact_search_probe:
            shadow_model = None
            shadow_metadata: dict[str, Any] = {}
            if mctx_lightzero_checkpoint_ref:
                shadow_model, shadow_metadata = _build_mctx_lightzero_shadow_model(
                    checkpoint_ref=mctx_lightzero_checkpoint_ref,
                    state_key=mctx_lightzero_checkpoint_state_key or None,
                    seed=int(checked["seed"]),
                    num_simulations=mctx_num_simulations,
                    batch_size=int(checked["batch_size"]) * PLAYER_COUNT,
                )
            search_service = MctxCompactSearchServiceV1(
                num_simulations=mctx_num_simulations,
                seed=int(checked["seed"]),
                config=MctxCompactSearchConfig(
                    hidden_dim=mctx_hidden_dim,
                    visual_channels=mctx_visual_channels,
                    require_gpu_backend=mctx_require_gpu_backend,
                ),
                shadow_model=shadow_model,
                model_metadata=shadow_metadata,
            )
            search_lane = (
                "mctx_compact_search_service_lightzero_jax_shadow"
                if shadow_model is not None
                else "mctx_compact_search_service"
            )
            if mctx_compare_direct_ctree:
                direct_ctree_service = _build_checkpoint_direct_ctree_compact_service(
                    checkpoint_ref=mctx_lightzero_checkpoint_ref,
                    state_key=mctx_lightzero_checkpoint_state_key or None,
                    seed=int(checked["seed"]),
                    num_simulations=mctx_num_simulations,
                    batch_size=int(checked["batch_size"]) * PLAYER_COUNT,
                    use_cuda=lightzero_consumer_use_cuda,
                    arrays_boundary_impl=mctx_compare_direct_ctree_impl,
                    input_mode=LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8,
                    temperature=1.0,
                    epsilon=0.0,
                )
                search_service = CompactSearchComparatorServiceV1(
                    primary=search_service,
                    reference=direct_ctree_service,
                    comparison_label=(f"{search_lane}_vs_{mctx_compare_direct_ctree_impl}"),
                )
                search_lane = f"{search_lane}:compare:{mctx_compare_direct_ctree_impl}"
        else:
            raise ValueError("hybrid_compact_rollout_slab_probe requires a search-service mode")
        if compact_root_tape_compare:
            compact_root_tape_recorder = InMemoryCompactRootTapeRecorderV1(
                tape_label=f"hybrid_profile:{search_lane}",
                allow_resident_host_snapshot=compact_root_tape_allow_resident_host_snapshot,
                max_records=compact_root_tape_max_records,
            )
            compact_root_tape_services["primary"] = search_service
            if compact_root_tape_compare_fixed_shape_floor:
                compact_root_tape_services["fixed_shape_floor"] = FixedShapeBatchedSearchOwnerV0(
                    root_count=int(checked["batch_size"]) * PLAYER_COUNT,
                    num_simulations=(
                        mctx_num_simulations
                        if mctx_compact_search_probe
                        else lightzero_consumer_num_simulations
                    ),
                )
            if compact_root_tape_compare_mctx:
                compact_root_tape_services["mctx"] = MctxCompactSearchServiceV1(
                    num_simulations=mctx_num_simulations,
                    seed=int(checked["seed"]),
                    config=MctxCompactSearchConfig(
                        hidden_dim=mctx_hidden_dim,
                        visual_channels=mctx_visual_channels,
                        require_all_roots_active=False,
                        require_gpu_backend=mctx_require_gpu_backend,
                    ),
                )
            if compact_root_tape_compare_model_compile:
                if not isinstance(batched_stack_probe, _LightZeroArrayCeilingStackProbe):
                    raise ValueError(
                        "model-compile root-tape comparison requires the compact "
                        "Torch array-ceiling probe"
                    )
                mode_label = compact_root_tape_model_compile_mode.replace("-", "_")
                compact_root_tape_services[f"model_compile_{mode_label}"] = (
                    batched_stack_probe.new_compact_torch_search_service_variant(
                        request_model_compile=True,
                        require_model_compile=compact_root_tape_require_model_compile,
                        model_compile_mode=compact_root_tape_model_compile_mode,
                    )
                )
            if compact_root_tape_compare_direct_core:
                if not isinstance(batched_stack_probe, _LightZeroArrayCeilingStackProbe):
                    raise ValueError(
                        "direct-core root-tape comparison requires the compact "
                        "Torch array-ceiling probe"
                    )
                compact_root_tape_services["initial_inference_direct_core"] = (
                    batched_stack_probe.new_compact_torch_search_service_variant(
                        request_model_compile=False,
                        require_model_compile=False,
                        model_compile_mode="reduce-overhead",
                        initial_inference_mode=(COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE),
                    )
                )
        compact_rollout_slab = CompactRolloutSlab(
            batch_size=int(checked["batch_size"]),
            player_count=PLAYER_COUNT,
            search_service=search_service,
            search_lane=f"compact_rollout_slab:{search_lane}",
            policy_source=f"hybrid_profile_compact_rollout_slab:{search_lane}",
            copy_root_observation=False,
            action_feedback_mode=compact_rollout_slab_action_mode,
            root_tape_recorder=compact_root_tape_recorder,
        )
        manager_batched_stack_probe = None
    profile = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=int(checked["batch_size"]),
            actor_count=actor_count,
            player_count=PLAYER_COUNT,
            steps=int(checked["steps"]),
            warmup_steps=int(checked["warmup_steps"]),
            seed=int(checked["seed"]),
            max_ticks=int(checked["max_ticks"]),
            body_capacity=int(checked["body_capacity"]),
            death_mode=str(checked["death_mode"]),
            decision_source_frames=1,
            source_physics_step_ms=SOURCE_PHYSICS_STEP_MS,
            pickle_payload=bool(checked["pickle_lightzero_payload"]),
            stack_storage_dtype=stack_storage_dtype,
            update_host_observation_stack=not hybrid_device_only_stack,
            refresh_observation_stack=hybrid_refresh_observation_stack,
            resident_observation_search=hybrid_resident_observation_search,
            materialize_scalar_timestep=materialize_scalar_timestep,
            native_actor_buffer=hybrid_native_actor_buffer,
            persistent_compact_render_state_buffer=persistent_compact_render_state_buffer,
            borrow_single_actor_render_state=borrow_single_actor_render_state,
            compact_service_replay_proof=compact_service_replay_proof,
            compact_rollout_slab_sample_gate=compact_rollout_slab_sample_gate,
            compact_rollout_slab_sample_gate_batch_size=(
                compact_rollout_slab_sample_gate_batch_size
            ),
            compact_rollout_slab_sample_gate_interval=(compact_rollout_slab_sample_gate_interval),
            compact_rollout_slab_sample_gate_replay_pair_capacity=(
                compact_rollout_slab_sample_gate_replay_pair_capacity
            ),
            compact_rollout_slab_learner_gate=compact_rollout_slab_learner_gate,
            compact_rollout_slab_learner_gate_train_steps=(
                compact_rollout_slab_learner_gate_train_steps
            ),
            compact_rollout_slab_learner_gate_device=(compact_rollout_slab_learner_gate_device),
            compact_rollout_slab_learner_gate_include_rnd=(
                compact_rollout_slab_learner_gate_include_rnd
            ),
            compact_rollout_slab_learner_gate_impl=(compact_rollout_slab_learner_gate_impl),
            compact_rollout_slab_learner_gate_support_scale=(
                compact_rollout_slab_learner_gate_support_scale
            ),
            compact_rollout_slab_learner_gate_num_unroll_steps=(
                compact_rollout_slab_learner_gate_num_unroll_steps
            ),
            compact_rollout_slab_action_mode=compact_rollout_slab_action_mode,
            compact_owned_loop_entrypoint=compact_owned_loop_entrypoint,
            compact_owned_loop_policy_version_ref=(compact_owned_loop_policy_version_ref),
            compact_owned_loop_model_version_ref=compact_owned_loop_model_version_ref,
            compact_owned_loop_policy_source=compact_owned_loop_policy_source,
            compact_owned_loop_capture_replay_store_state=(
                compact_owned_loop_capture_replay_store_state
            ),
        ),
        observation_renderer=renderer,
        policy_search_probe=policy_search_probe,
        batched_stack_probe=manager_batched_stack_probe,
        compact_rollout_slab=compact_rollout_slab,
    )
    compact_root_tape_report = None
    compact_root_tape_error = ""
    if compact_root_tape_recorder is not None:
        tape = compact_root_tape_recorder.build_tape(
            metadata={
                "profile_only": True,
                "calls_train_muzero": False,
                "action_mode": compact_rollout_slab_action_mode,
                "root_noise_weight": (
                    -1.0
                    if lightzero_consumer_root_noise_weight is None
                    else lightzero_consumer_root_noise_weight
                ),
            }
        )
        compact_root_tape_report = run_compact_root_tape_comparison_v1(
            tape,
            services=compact_root_tape_services,
            reference_label=compact_root_tape_reference_label,
        )
    if lightzero_collect_forward_probe:
        lightzero_probe_gap = (
            "LightZero collect-forward probe uses real public collect_mode.forward "
            "but LightZero MCTS/tree internals still cross to CPU"
        )
    elif lightzero_initial_inference_probe:
        lightzero_probe_gap = (
            "LightZero initial-inference probe calls the real MuZero model only; "
            "it deliberately excludes MCTS/tree search"
        )
    elif lightzero_array_ceiling_probe:
        if lightzero_array_ceiling_mode in LIGHTZERO_ARRAY_CEILING_DENSE_TORCH_MCTS_MODES:
            lightzero_probe_gap = (
                "LightZero dense_torch_mcts probe uses real MuZero model calls and "
                "a profile-only GPU tensor PUCT search; it is not stock LightZero CTree"
            )
        else:
            lightzero_probe_gap = (
                "LightZero array-ceiling probe uses real MuZero model calls "
                "with compact arrays out; it is not real MCTS"
            )
    elif mctx_compact_search_probe:
        lightzero_probe_gap = (
            "MCTX compact search probe runs a profile-only JAX/MCTX Gumbel MuZero "
            "search behind CompactSearchServiceV1; it is not stock LightZero CTree "
            "and not train_muzero"
        )
        if mctx_compare_direct_ctree:
            lightzero_probe_gap += (
                "; same-root direct CTree comparison telemetry is enabled, but "
                "MCTX remains the primary action source for this profile row"
            )
        if compact_root_tape_compare:
            lightzero_probe_gap += (
                "; fixed-root tape replay is enabled for bounded same-root "
                "comparison, but this does not make MCTX a training backend"
            )
    elif lightzero_mcts_arrays_boundary_probe and lightzero_mcts_arrays_boundary_impl in {
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT,
    }:
        lightzero_probe_gap = (
            "LightZero MCTS arrays-boundary direct CTree calls the real MuZero "
            "model and real CTree MCTS directly; it bypasses public "
            "collect_mode.forward and remains parity-gated"
        )
    elif lightzero_mcts_arrays_boundary_probe:
        lightzero_probe_gap = (
            "LightZero MCTS arrays-boundary facade calls stock collect_mode.forward "
            "and returns compact arrays"
        )
    else:
        lightzero_probe_gap = "batched stack probe is synthetic and not LightZero MCTS"
    resident_reuse_gap = []
    if (
        lightzero_mcts_arrays_boundary_input_mode
        == LIGHTZERO_ARRAY_CEILING_INPUT_MODE_RESIDENT_TORCH_REUSE
        or lightzero_array_ceiling_input_mode
        == LIGHTZERO_ARRAY_CEILING_INPUT_MODE_RESIDENT_TORCH_REUSE
    ):
        resident_reuse_gap = [
            "resident_torch_reuse reuses a stale first input tensor after warmup; it is an upper-bound profile ceiling, not a valid training input path"
        ]

    result = {
        **profile,
        "ok": True,
        "impl_id": "curvyzero_modal_profile_only_hybrid_actor_gpu_renderer/v0",
        "hybrid_observation_canary": True,
        "profile_only": True,
        "calls_train_muzero": False,
        "stock_lightzero_integrated": False,
        "touches_live_runs": False,
        "trainer_defaults_changed": False,
        "max_ticks": int(checked["max_ticks"]),
        "app_name": APP_NAME,
        "renderer_app_name": RENDER_APP_NAME,
        "config": {
            **checked,
            "actor_count": actor_count,
            "hybrid_policy_probe_simulations": policy_probe_simulations,
            "hybrid_policy_probe_channels": policy_probe_channels,
            "hybrid_batched_stack_probe_simulations": batched_stack_probe_simulations,
            "hybrid_batched_stack_probe_channels": batched_stack_probe_channels,
            "hybrid_batched_stack_probe_device_latest": device_latest_stack_probe,
            "hybrid_stack_storage_dtype": stack_storage_dtype,
            "hybrid_materialize_scalar_timestep": materialize_scalar_timestep,
            "hybrid_compact_service_replay_proof": compact_service_replay_proof,
            "hybrid_compact_rollout_slab_probe": compact_rollout_slab_probe,
            "hybrid_compact_rollout_slab_sample_gate": compact_rollout_slab_sample_gate,
            "hybrid_compact_rollout_slab_sample_gate_batch_size": (
                compact_rollout_slab_sample_gate_batch_size
            ),
            "hybrid_compact_rollout_slab_sample_gate_interval": (
                compact_rollout_slab_sample_gate_interval
            ),
            "hybrid_compact_rollout_slab_sample_gate_replay_pair_capacity": (
                compact_rollout_slab_sample_gate_replay_pair_capacity
            ),
            "hybrid_compact_rollout_slab_learner_gate": compact_rollout_slab_learner_gate,
            "hybrid_compact_rollout_slab_learner_gate_train_steps": (
                compact_rollout_slab_learner_gate_train_steps
            ),
            "hybrid_compact_rollout_slab_learner_gate_device": (
                compact_rollout_slab_learner_gate_device
            ),
            "hybrid_compact_rollout_slab_learner_gate_include_rnd": (
                compact_rollout_slab_learner_gate_include_rnd
            ),
            "hybrid_compact_rollout_slab_learner_gate_impl": (
                compact_rollout_slab_learner_gate_impl
            ),
            "hybrid_compact_rollout_slab_learner_gate_support_scale": (
                compact_rollout_slab_learner_gate_support_scale
            ),
            "hybrid_compact_rollout_slab_learner_gate_num_unroll_steps": (
                compact_rollout_slab_learner_gate_num_unroll_steps
            ),
            "hybrid_compact_rollout_slab_action_mode": compact_rollout_slab_action_mode,
            "hybrid_compact_owned_loop_entrypoint": compact_owned_loop_entrypoint,
            "hybrid_compact_owned_loop_policy_version_ref": (compact_owned_loop_policy_version_ref),
            "hybrid_compact_owned_loop_model_version_ref": (compact_owned_loop_model_version_ref),
            "hybrid_compact_owned_loop_policy_source": compact_owned_loop_policy_source,
            "hybrid_compact_owned_loop_capture_replay_store_state": (
                compact_owned_loop_capture_replay_store_state
            ),
            "hybrid_compact_root_tape_compare": compact_root_tape_compare,
            "hybrid_compact_root_tape_max_records": compact_root_tape_max_records,
            "hybrid_compact_root_tape_allow_resident_host_snapshot": (
                compact_root_tape_allow_resident_host_snapshot
            ),
            "hybrid_compact_root_tape_compare_fixed_shape_floor": (
                compact_root_tape_compare_fixed_shape_floor
            ),
            "hybrid_compact_root_tape_compare_mctx": compact_root_tape_compare_mctx,
            "hybrid_compact_root_tape_compare_model_compile": (
                compact_root_tape_compare_model_compile
            ),
            "hybrid_compact_root_tape_compare_direct_core": (compact_root_tape_compare_direct_core),
            "hybrid_compact_root_tape_model_compile_mode": (compact_root_tape_model_compile_mode),
            "hybrid_compact_root_tape_require_model_compile": (
                compact_root_tape_require_model_compile
            ),
            "hybrid_compact_root_tape_reference_label": compact_root_tape_reference_label,
            "hybrid_resident_chunk_probe": resident_chunk_probe,
            "hybrid_resident_replay_steps": resident_replay_steps,
            "hybrid_resident_sample_batch_size": resident_sample_batch_size,
            "hybrid_resident_replay_train_steps": resident_replay_train_steps,
            "hybrid_resident_readback_checksum": resident_readback_checksum,
            "hybrid_lightzero_collect_forward_probe": lightzero_collect_forward_probe,
            "hybrid_lightzero_initial_inference_probe": lightzero_initial_inference_probe,
            "hybrid_lightzero_array_ceiling_probe": lightzero_array_ceiling_probe,
            "hybrid_lightzero_mcts_arrays_boundary_probe": (lightzero_mcts_arrays_boundary_probe),
            "hybrid_mctx_compact_search_probe": mctx_compact_search_probe,
            "hybrid_lightzero_mcts_arrays_boundary_impl": (lightzero_mcts_arrays_boundary_impl),
            "hybrid_lightzero_mcts_arrays_boundary_input_mode": (
                lightzero_mcts_arrays_boundary_input_mode
            ),
            "hybrid_lightzero_array_ceiling_mode": lightzero_array_ceiling_mode,
            "hybrid_lightzero_array_ceiling_input_mode": (lightzero_array_ceiling_input_mode),
            "hybrid_lightzero_consumer_num_simulations": lightzero_consumer_num_simulations,
            "hybrid_lightzero_consumer_temperature": lightzero_consumer_temperature,
            "hybrid_lightzero_consumer_epsilon": lightzero_consumer_epsilon,
            "hybrid_lightzero_consumer_root_noise_weight": (
                -1.0
                if lightzero_consumer_root_noise_weight is None
                else lightzero_consumer_root_noise_weight
            ),
            "hybrid_lightzero_consumer_use_cuda": lightzero_consumer_use_cuda,
            "hybrid_lightzero_consumer_collect_with_pure_policy": (
                lightzero_consumer_collect_with_pure_policy
            ),
            "hybrid_compact_torch_compile_model_inference": (compact_torch_compile_model_inference),
            "hybrid_compact_torch_compile_search": compact_torch_compile_search,
            "hybrid_compact_torch_require_model_compile": compact_torch_require_model_compile,
            "hybrid_compact_torch_model_compile_mode": (compact_torch_model_compile_mode),
            "hybrid_compact_torch_recurrent_action_shape_mode": (
                compact_torch_recurrent_action_shape_mode
            ),
            "hybrid_compact_torch_timing_mode": compact_torch_timing_mode,
            "hybrid_compact_torch_initial_inference_mode": (compact_torch_initial_inference_mode),
            "hybrid_compact_torch_observation_memory_format": (
                compact_torch_observation_memory_format
            ),
            "hybrid_compact_torch_model_memory_format": (compact_torch_model_memory_format),
            "hybrid_mctx_num_simulations": mctx_num_simulations,
            "hybrid_mctx_hidden_dim": mctx_hidden_dim,
            "hybrid_mctx_visual_channels": mctx_visual_channels,
            "hybrid_mctx_require_gpu_backend": mctx_require_gpu_backend,
            "hybrid_mctx_lightzero_checkpoint_ref": mctx_lightzero_checkpoint_ref,
            "hybrid_mctx_lightzero_checkpoint_state_key": mctx_lightzero_checkpoint_state_key,
            "hybrid_mctx_compare_direct_ctree": mctx_compare_direct_ctree,
            "hybrid_mctx_compare_direct_ctree_impl": mctx_compare_direct_ctree_impl,
        },
        "jax": {
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_count": len(jax.devices()),
        },
        "nvidia_smi": _nvidia_smi(),
        "packages": _package_versions(),
        "known_gaps": [
            *_known_gaps(),
            "hybrid observation canary only; does not call stock train_muzero",
            "terminal final_observation is modeled for the profile scaffold, but natural-death trainer semantics are not claimed",
            "actors are in-process; subprocess IPC/fan-in is not measured here",
            lightzero_probe_gap,
            *resident_reuse_gap,
            "device-latest stack probe is no-death/profile-only; terminal reset semantics are not claimed",
        ],
    }
    if compact_root_tape_recorder is not None:
        result.update(
            {
                "compact_root_tape_compare_enabled": True,
                "compact_root_tape_record_count": compact_root_tape_recorder.record_count,
                "compact_root_tape_skipped_record_count": (
                    compact_root_tape_recorder.skipped_record_count
                ),
                "compact_root_tape_reference_label": compact_root_tape_reference_label,
                "compact_root_tape_service_labels": sorted(compact_root_tape_services),
                "compact_root_tape_comparison": compact_root_tape_report,
                "compact_root_tape_error": compact_root_tape_error,
            }
        )
    else:
        result.update(
            {
                "compact_root_tape_compare_enabled": False,
                "compact_root_tape_record_count": 0,
                "compact_root_tape_skipped_record_count": 0,
                "compact_root_tape_reference_label": "",
                "compact_root_tape_service_labels": [],
                "compact_root_tape_comparison": None,
                "compact_root_tape_error": "",
            }
        )
    if bool(config.get("compact_hybrid_output", True)):
        return _compact_hybrid_observation_profile_result(result)
    return result


def _compact_hybrid_observation_profile_result(result: dict[str, Any]) -> dict[str, Any]:
    policy_env_id = [int(value) for value in result.get("last_policy_env_id", [])]
    policy_env_row = [int(value) for value in result.get("last_policy_env_row", [])]
    policy_player = [int(value) for value in result.get("last_policy_player", [])]
    payload_summary = dict(result.get("last_payload_summary") or {})
    return {
        "schema_id": result.get("schema_id"),
        "impl_id": result.get("impl_id"),
        "ok": result.get("ok"),
        "profile_only": result.get("profile_only"),
        "hybrid_observation_canary": result.get("hybrid_observation_canary"),
        "calls_train_muzero": result.get("calls_train_muzero"),
        "stock_lightzero_integrated": result.get("stock_lightzero_integrated"),
        "trainer_defaults_changed": result.get("trainer_defaults_changed"),
        "touches_live_runs": result.get("touches_live_runs"),
        "app_name": result.get("app_name"),
        "renderer_app_name": result.get("renderer_app_name"),
        "observation_mode": result.get("observation_mode"),
        "renderer_backend_name": result.get("renderer_backend_name"),
        "stack_storage_dtype": result.get("stack_storage_dtype"),
        "materialize_scalar_timestep": result.get("materialize_scalar_timestep"),
        "compact_owned_loop_entrypoint_enabled": result.get(
            "compact_owned_loop_entrypoint_enabled"
        ),
        "compact_owned_loop_schema_id": result.get("compact_owned_loop_schema_id"),
        "compact_owned_loop_profile_only": result.get("compact_owned_loop_profile_only"),
        "compact_owned_loop_calls_train_muzero": result.get(
            "compact_owned_loop_calls_train_muzero"
        ),
        "compact_owned_loop_touches_live_runs": result.get("compact_owned_loop_touches_live_runs"),
        "compact_owned_loop_replay_store_owned": result.get(
            "compact_owned_loop_replay_store_owned"
        ),
        "compact_owned_loop_policy_version_handoff": result.get(
            "compact_owned_loop_policy_version_handoff"
        ),
        "compact_owned_loop_policy_version_ref": result.get(
            "compact_owned_loop_policy_version_ref"
        ),
        "compact_owned_loop_model_version_ref": result.get("compact_owned_loop_model_version_ref"),
        "compact_owned_loop_policy_source": result.get("compact_owned_loop_policy_source"),
        "compact_owned_loop_telemetry": result.get("compact_owned_loop_telemetry"),
        "compact_owned_loop_replay_store_state_metadata": result.get(
            "compact_owned_loop_replay_store_state_metadata"
        ),
        "compact_rollout_slab_enabled": result.get("compact_rollout_slab_enabled"),
        "compact_rollout_slab_profile_only": result.get("compact_rollout_slab_profile_only"),
        "compact_rollout_slab_calls": result.get("compact_rollout_slab_calls"),
        "compact_rollout_slab_total_roots": result.get("compact_rollout_slab_total_roots"),
        "compact_rollout_slab_roots_per_call": result.get("compact_rollout_slab_roots_per_call"),
        "compact_rollout_slab_committed_index_row_count": result.get(
            "compact_rollout_slab_committed_index_row_count"
        ),
        "compact_rollout_slab_action_mode": result.get("compact_rollout_slab_action_mode"),
        "compact_rollout_slab_action_override_drop_count": result.get(
            "compact_rollout_slab_action_override_drop_count"
        ),
        "compact_rollout_slab_telemetry_totals": result.get(
            "compact_rollout_slab_telemetry_totals"
        ),
        "compact_rollout_slab_last_telemetry": result.get("compact_rollout_slab_last_telemetry"),
        "compact_rollout_slab_sample_gate_enabled": result.get(
            "compact_rollout_slab_sample_gate_enabled"
        ),
        "compact_rollout_slab_sample_gate_calls": result.get(
            "compact_rollout_slab_sample_gate_calls"
        ),
        "compact_rollout_slab_sample_gate_index_row_count": result.get(
            "compact_rollout_slab_sample_gate_index_row_count"
        ),
        "compact_rollout_slab_sample_gate_target_row_count": result.get(
            "compact_rollout_slab_sample_gate_target_row_count"
        ),
        "compact_rollout_slab_sample_gate_sample_row_count": result.get(
            "compact_rollout_slab_sample_gate_sample_row_count"
        ),
        "compact_rollout_slab_sample_gate_batch_size": result.get(
            "compact_rollout_slab_sample_gate_batch_size"
        ),
        "compact_rollout_slab_sample_gate_interval": result.get(
            "compact_rollout_slab_sample_gate_interval"
        ),
        "compact_rollout_slab_sample_gate_replay_ring_pair_capacity": result.get(
            "compact_rollout_slab_sample_gate_replay_ring_pair_capacity"
        ),
        "compact_rollout_slab_sample_gate_replay_ring_entry_count": result.get(
            "compact_rollout_slab_sample_gate_replay_ring_entry_count"
        ),
        "compact_rollout_slab_sample_gate_replay_ring_index_row_count": result.get(
            "compact_rollout_slab_sample_gate_replay_ring_index_row_count"
        ),
        "compact_rollout_slab_sample_gate_replay_ring_evicted_pair_count": result.get(
            "compact_rollout_slab_sample_gate_replay_ring_evicted_pair_count"
        ),
        "compact_rollout_slab_sample_gate_replay_ring_evicted_index_row_count": (
            result.get("compact_rollout_slab_sample_gate_replay_ring_evicted_index_row_count")
        ),
        "compact_rollout_slab_sample_gate_opportunities": result.get(
            "compact_rollout_slab_sample_gate_opportunities"
        ),
        "compact_rollout_slab_sample_gate_skipped_count": result.get(
            "compact_rollout_slab_sample_gate_skipped_count"
        ),
        "compact_rollout_slab_sample_gate_sec": result.get("compact_rollout_slab_sample_gate_sec"),
        "compact_rollout_slab_sample_gate_mock_base_env_timestep_rows": result.get(
            "compact_rollout_slab_sample_gate_mock_base_env_timestep_rows"
        ),
        "compact_rollout_slab_sample_gate_last_telemetry": result.get(
            "compact_rollout_slab_sample_gate_last_telemetry"
        ),
        "compact_rollout_slab_learner_gate_enabled": result.get(
            "compact_rollout_slab_learner_gate_enabled"
        ),
        "compact_rollout_slab_learner_gate_calls": result.get(
            "compact_rollout_slab_learner_gate_calls"
        ),
        "compact_rollout_slab_learner_gate_updates": result.get(
            "compact_rollout_slab_learner_gate_updates"
        ),
        "compact_rollout_slab_learner_gate_sample_row_count": result.get(
            "compact_rollout_slab_learner_gate_sample_row_count"
        ),
        "compact_rollout_slab_learner_gate_input_bytes": result.get(
            "compact_rollout_slab_learner_gate_input_bytes"
        ),
        "compact_rollout_slab_learner_gate_sec": result.get(
            "compact_rollout_slab_learner_gate_sec"
        ),
        "compact_rollout_slab_learner_gate_train_steps": result.get(
            "compact_rollout_slab_learner_gate_train_steps"
        ),
        "compact_rollout_slab_learner_gate_device": result.get(
            "compact_rollout_slab_learner_gate_device"
        ),
        "compact_rollout_slab_learner_gate_include_rnd": result.get(
            "compact_rollout_slab_learner_gate_include_rnd"
        ),
        "compact_rollout_slab_learner_gate_impl": result.get(
            "compact_rollout_slab_learner_gate_impl"
        ),
        "compact_rollout_slab_learner_gate_toy_probe": result.get(
            "compact_rollout_slab_learner_gate_toy_probe"
        ),
        "compact_rollout_slab_learner_gate_real_muzero_update": result.get(
            "compact_rollout_slab_learner_gate_real_muzero_update"
        ),
        "compact_rollout_slab_learner_gate_support_scale": result.get(
            "compact_rollout_slab_learner_gate_support_scale"
        ),
        "compact_rollout_slab_learner_gate_num_unroll_steps": result.get(
            "compact_rollout_slab_learner_gate_num_unroll_steps"
        ),
        "compact_rollout_slab_learner_gate_last_telemetry": result.get(
            "compact_rollout_slab_learner_gate_last_telemetry"
        ),
        "compact_service_replay_proof_enabled": result.get("compact_service_replay_proof_enabled"),
        "compact_service_replay_proof_calls": result.get("compact_service_replay_proof_calls"),
        "compact_service_replay_proof_skipped_count": result.get(
            "compact_service_replay_proof_skipped_count"
        ),
        "compact_service_replay_proof_target_row_count": result.get(
            "compact_service_replay_proof_target_row_count"
        ),
        "compact_service_replay_proof_warmup_seeded_calls": result.get(
            "compact_service_replay_proof_warmup_seeded_calls"
        ),
        "compact_service_replay_proof_warmup_seeded_target_row_count": result.get(
            "compact_service_replay_proof_warmup_seeded_target_row_count"
        ),
        "compact_service_replay_proof_warmup_seeded_sec": result.get(
            "compact_service_replay_proof_warmup_seeded_sec"
        ),
        "compact_service_replay_proof_sec": result.get("compact_service_replay_proof_sec"),
        "compact_service_replay_proof_sec_per_call": result.get(
            "compact_service_replay_proof_sec_per_call"
        ),
        "compact_service_replay_proof_sec_per_target_row": result.get(
            "compact_service_replay_proof_sec_per_target_row"
        ),
        "compact_service_replay_proof_last_telemetry": result.get(
            "compact_service_replay_proof_last_telemetry"
        ),
        "compact_root_tape_compare_enabled": result.get("compact_root_tape_compare_enabled"),
        "compact_root_tape_record_count": result.get("compact_root_tape_record_count"),
        "compact_root_tape_skipped_record_count": result.get(
            "compact_root_tape_skipped_record_count"
        ),
        "compact_root_tape_reference_label": result.get("compact_root_tape_reference_label"),
        "compact_root_tape_service_labels": result.get("compact_root_tape_service_labels"),
        "compact_root_tape_comparison": result.get("compact_root_tape_comparison"),
        "compact_root_tape_error": result.get("compact_root_tape_error"),
        "policy_search_probe_backend_name": result.get("policy_search_probe_backend_name"),
        "policy_search_probe_semantics": result.get("policy_search_probe_semantics"),
        "policy_search_probe_calls": result.get("policy_search_probe_calls"),
        "policy_search_probe_total_roots": result.get("policy_search_probe_total_roots"),
        "policy_search_probe_roots_per_call": result.get("policy_search_probe_roots_per_call"),
        "policy_search_probe_input_shape": result.get("policy_search_probe_input_shape"),
        "policy_search_probe_input_dtype": result.get("policy_search_probe_input_dtype"),
        "policy_search_probe_input_bytes_total": result.get(
            "policy_search_probe_input_bytes_total"
        ),
        "policy_search_probe_last_telemetry": result.get("policy_search_probe_last_telemetry"),
        "batched_stack_probe_backend_name": result.get("batched_stack_probe_backend_name"),
        "batched_stack_probe_semantics": result.get("batched_stack_probe_semantics"),
        "batched_stack_probe_calls": result.get("batched_stack_probe_calls"),
        "batched_stack_probe_total_roots": result.get("batched_stack_probe_total_roots"),
        "batched_stack_probe_roots_per_call": result.get("batched_stack_probe_roots_per_call"),
        "batched_stack_probe_input_shape": result.get("batched_stack_probe_input_shape"),
        "batched_stack_probe_input_dtype": result.get("batched_stack_probe_input_dtype"),
        "batched_stack_probe_input_bytes_total": result.get(
            "batched_stack_probe_input_bytes_total"
        ),
        "batched_stack_probe_last_telemetry": result.get("batched_stack_probe_last_telemetry"),
        "batch_size": result.get("batch_size"),
        "actor_count": result.get("actor_count"),
        "player_count": result.get("player_count"),
        "death_mode": result.get("death_mode"),
        "done_semantics_verified": result.get("done_semantics_verified"),
        "terminated_row_count": result.get("terminated_row_count"),
        "truncated_row_count": result.get("truncated_row_count"),
        "death_row_count": result.get("death_row_count"),
        "death_count_total": result.get("death_count_total"),
        "death_cause_count_by_name": result.get("death_cause_count_by_name"),
        "normal_collision_death_causes": result.get("normal_collision_death_causes"),
        "normal_collision_death_hit_owner_present": result.get(
            "normal_collision_death_hit_owner_present"
        ),
        "normal_collision_death_evidence_rows": result.get("normal_collision_death_evidence_rows"),
        "terminal_final_observation_row_count": result.get("terminal_final_observation_row_count"),
        "terminal_final_observation_before_autoreset_verified": result.get(
            "terminal_final_observation_before_autoreset_verified"
        ),
        "terminal_final_reward_map_row_count": result.get("terminal_final_reward_map_row_count"),
        "terminal_final_reward_map_matches_reward_row_count": result.get(
            "terminal_final_reward_map_matches_reward_row_count"
        ),
        "terminal_final_reward_map_verified": result.get("terminal_final_reward_map_verified"),
        "steps": result.get("steps"),
        "warmup_steps": result.get("warmup_steps"),
        "max_ticks": result.get("max_ticks"),
        "env_action_checksum_total": result.get("env_action_checksum_total"),
        "env_done_checksum_total": result.get("env_done_checksum_total"),
        "env_reward_checksum_total": result.get("env_reward_checksum_total"),
        "env_action_mask_checksum_total": result.get("env_action_mask_checksum_total"),
        "env_trajectory_checksum_total": result.get("env_trajectory_checksum_total"),
        "last_env_action_checksum": result.get("last_env_action_checksum"),
        "last_env_trajectory_checksum": result.get("last_env_trajectory_checksum"),
        "rows_per_step": result.get("rows_per_step"),
        "ready_count": result.get("ready_count"),
        "timestep_count": result.get("timestep_count"),
        "materialized_timestep_count": result.get("materialized_timestep_count"),
        "live_physical_row_count": result.get("live_physical_row_count"),
        "terminal_row_count": result.get("terminal_row_count"),
        "autoreset_row_count": result.get("autoreset_row_count"),
        "done_rows": result.get("done_rows"),
        "total_sec": result.get("total_sec"),
        "measured_sec": result.get("measured_sec"),
        "warmup_sec": result.get("warmup_sec"),
        "steps_per_sec": result.get("steps_per_sec"),
        "physical_rows_per_sec": result.get("physical_rows_per_sec"),
        "timings": result.get("timings"),
        "timing_per_timestep_sec": result.get("timing_per_timestep_sec"),
        "compact_payload_bytes_per_step": result.get("compact_payload_bytes_per_step"),
        "compact_payload_bytes_per_timestep": result.get("compact_payload_bytes_per_timestep"),
        "rendered_stack_bytes_per_step": result.get("rendered_stack_bytes_per_step"),
        "compact_vs_rendered_stack_ratio": result.get("compact_vs_rendered_stack_ratio"),
        "last_observation_shape": result.get("last_observation_shape"),
        "last_observation_dtype": result.get("last_observation_dtype"),
        "last_flat_obs_shape": result.get("last_flat_obs_shape"),
        "last_flat_obs_dtype": result.get("last_flat_obs_dtype"),
        "last_target_reward_shape": result.get("last_target_reward_shape"),
        "last_policy_env_id_head": policy_env_id[:16],
        "last_policy_env_row_head": policy_env_row[:16],
        "last_policy_player_head": policy_player[:16],
        "last_policy_env_id_tail": policy_env_id[-16:],
        "last_policy_env_row_tail": policy_env_row[-16:],
        "last_policy_player_tail": policy_player[-16:],
        "last_payload_summary": {
            "global_rows_head": list(payload_summary.get("global_rows", []))[:16],
            "global_rows_tail": list(payload_summary.get("global_rows", []))[-16:],
            "terminal_global_rows": payload_summary.get("terminal_global_rows", []),
            "autoreset_global_rows": payload_summary.get("autoreset_global_rows", []),
        },
        "jax": result.get("jax"),
        "nvidia_smi": result.get("nvidia_smi"),
        "packages": result.get("packages"),
        "contract": result.get("contract"),
        "config": result.get("config"),
        "known_gaps": result.get("known_gaps"),
    }


def _build_rnd_model(config: dict[str, Any]) -> xb.CurvyRNDRewardModel:
    if not bool(config["include_lightzero_payload_profile"]):
        raise ValueError("include_rnd_meter requires include_lightzero_payload_profile")
    return xb.CurvyRNDRewardModel(
        {
            "input_type": "obs",
            "batch_size": int(config["rnd_batch_size"]),
            "update_per_collect": int(config["rnd_update_per_collect"]),
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
            "intrinsic_reward_weight": 0.0,
            "seed": int(config["seed"]),
            "disable_cudnn": str(config["rnd_device"]).startswith("cuda"),
            "curvyzero_adapter": {
                "shape": xb.RND_INPUT_SHAPE_POLICY_GRAY64_LATEST_V0,
                "source_observation_shape": xb.RND_SOURCE_OBSERVATION_SHAPE,
            },
        },
        device=str(config["rnd_device"]),
    )


def _render_candidate_frames(
    *,
    jax: Any,
    np: Any,
    env: VectorMultiplayerEnv,
    config: dict[str, Any],
    render_fn_for_slots: Any,
) -> tuple[Any, dict[str, float], dict[str, float]]:
    return _render_candidate_frames_from_production_state(
        jax=jax,
        np=np,
        production_state=env.state,
        config=config,
        render_fn_for_slots=render_fn_for_slots,
    )


def _render_candidate_frames_from_production_state(
    *,
    jax: Any,
    np: Any,
    production_state: dict[str, Any],
    config: dict[str, Any],
    render_fn_for_slots: Any,
) -> tuple[Any, dict[str, float], dict[str, float]]:
    max_render_trail_slots = int(config["trail_slots"])
    compact_config = dict(config)
    compact_config["trail_slots"] = int(config.get("body_capacity", max_render_trail_slots))
    started = time.perf_counter()
    compact_state = _production_to_benchmark_source_state(
        np=np,
        production_state=production_state,
        config=compact_config,
    )
    production_to_compact_sec = time.perf_counter() - started

    started = time.perf_counter()
    compact_state = _pack_compact_trails_in_owner_draw_order(
        np=np,
        state=compact_state,
        config=compact_config,
    )
    owner_ordered_pack_sec = time.perf_counter() - started
    render_trail_slots = _select_render_trail_slots(
        np=np,
        compact_state=compact_state,
        config=config,
    )
    trail_stats = _trail_stats_after_owner_pack(
        np=np,
        compact_state=compact_state,
        render_trail_slots=render_trail_slots,
        max_render_trail_slots=max_render_trail_slots,
    )
    _assert_no_render_truncation_if_required(config=config, trail_stats=trail_stats)
    compact_state = _truncate_compact_trails_for_render(
        np=np,
        compact_state=compact_state,
        render_trail_slots=render_trail_slots,
    )

    started = time.perf_counter()
    device_state = _copy_state_to_device(jax=jax, state=compact_state)
    host_to_device_sec = time.perf_counter() - started

    started = time.perf_counter()
    render_fn = render_fn_for_slots(render_trail_slots)
    output_device = render_fn(device_state)
    output_device.block_until_ready()
    device_render_sec = time.perf_counter() - started

    started = time.perf_counter()
    view_major = np.asarray(output_device)
    device_to_host_sec = time.perf_counter() - started

    started = time.perf_counter()
    row_major = _view_major_to_row_major_frames(
        view_major,
        batch_size=int(config["batch_size"]),
    )
    view_major_to_row_major_sec = time.perf_counter() - started

    return (
        row_major,
        {
            "production_to_compact_sec": production_to_compact_sec,
            "owner_ordered_pack_sec": owner_ordered_pack_sec,
            "host_to_device_sec": host_to_device_sec,
            "device_render_sec": device_render_sec,
            "device_to_host_sec": device_to_host_sec,
            "view_major_to_row_major_sec": view_major_to_row_major_sec,
        },
        trail_stats,
    )


def _prewarm_dynamic_render_functions(
    *,
    jax: Any,
    np: Any,
    production_state: dict[str, Any],
    config: dict[str, Any],
    render_fn_for_slots: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    slots = _render_prewarm_slots(config)
    slot_secs: dict[str, float] = {}
    for slot in slots:
        slot_config = dict(config)
        slot_config["trail_slots"] = int(slot)
        slot_config["max_render_trail_slots"] = int(slot)
        slot_config["dynamic_render_trail_slots"] = False
        slot_started = time.perf_counter()
        _render_candidate_frames_from_production_state(
            jax=jax,
            np=np,
            production_state=production_state,
            config=slot_config,
            render_fn_for_slots=render_fn_for_slots,
        )
        slot_secs[str(slot)] = time.perf_counter() - slot_started
    return {
        "enabled": True,
        "slots": slots,
        "slot_sec": slot_secs,
        "total_sec": time.perf_counter() - started,
    }


def _render_prewarm_slots(config: dict[str, Any]) -> list[int]:
    max_slots = int(config["trail_slots"])
    if not bool(config.get("dynamic_render_trail_slots", False)):
        return [max_slots]
    min_slots = int(config.get("min_render_trail_slots", max_slots))
    slots: list[int] = []
    current = max(1, min_slots)
    while current < max_slots:
        slots.append(current)
        current *= 2
    slots.append(max_slots)
    return sorted(set(int(slot) for slot in slots))


class _PersistentJaxPolicyFramebufferRenderer:
    backend_name = SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND

    def __init__(
        self,
        *,
        jax: Any,
        jnp: Any,
        np: Any,
        config: dict[str, Any],
        bonus_render_mode_id: int,
    ) -> None:
        self._jax = jax
        self._jnp = jnp
        self._np = np
        self._config = config
        self._bonus_render_mode_id = int(bonus_render_mode_id)
        self._batch_size = int(config["batch_size"])
        self._target_size = TARGET_SIZE
        self._map_size = float(config["map_size"])
        self._bonus_count = int(config.get("bonus_count", 0))
        self._async_device_only_profile = bool(config.get("async_device_only_profile", False))
        self._vectorized_delta_pack_profile = bool(
            config.get("persistent_vectorized_delta_pack_profile", False)
        )
        self._layer_device: Any | None = None
        self._previous_cursor: Any | None = None
        self._previous_owner_pos: Any | None = None
        self._previous_owner_valid: Any | None = None
        self._previous_avatar_color: Any | None = None
        self._update_fn_cache: dict[int, Any] = {}
        self.last_output_device: Any | None = None
        self._compose_fn = _make_persistent_policy_compose_fn(
            jax=jax,
            jnp=jnp,
            target_size=self._target_size,
            map_size=self._map_size,
            bonus_count=self._bonus_count,
            bonus_render_mode_id=self._bonus_render_mode_id,
        )

    def render(self, request: SourceStateBatchedRenderRequest) -> SourceStateBatchedRenderResult:
        np = self._np
        rows = np.asarray(request.row_indices, dtype=np.int64)
        players = np.asarray(request.controlled_players, dtype=np.int64)
        out = np.asarray(request.out)
        self._validate_full_row_major_request(rows=rows, players=players, out=out)

        timings: dict[str, float] = {}
        started = time.perf_counter()
        request_state = dict(request.state)
        state_row_overlays = tuple(request.state_row_overlays)
        if state_row_overlays:
            if _is_persistent_compact_render_state(request_state):
                raise ValueError(
                    "persistent compact render-state requests cannot include row overlays"
                )
            request_state = source_state_render_state_with_row_overlays(
                request_state,
                state_row_overlays,
                batch_size=self._batch_size,
            )
        if _is_persistent_compact_render_state(request_state):
            compact_state = _validate_persistent_compact_render_state(
                np=np,
                state=request_state,
                config=self._config,
            )
            timings["persistent_compact_state_handoff_sec"] = time.perf_counter() - started
            timings["production_to_compact_sec"] = 0.0
            compact_state_passthrough = 1.0
        else:
            compact_state = _persistent_compact_state_from_production(
                np=np,
                production_state=request_state,
                config=self._config,
            )
            timings["production_to_compact_sec"] = time.perf_counter() - started
            timings["persistent_compact_state_handoff_sec"] = 0.0
            compact_state_passthrough = 0.0

        avatar_color = np.asarray(compact_state["avatar_color"], dtype=np.int32)
        if self._previous_avatar_color is not None and not bool(
            np.array_equal(avatar_color, self._previous_avatar_color)
        ):
            self._previous_cursor = None
            self._previous_owner_pos = None
            self._previous_owner_valid = None

        started = time.perf_counter()
        delta_state, next_cursor, delta_stats = _persistent_delta_state(
            np=np,
            compact_state=compact_state,
            previous_cursor=self._previous_cursor,
            previous_owner_pos=self._previous_owner_pos,
            previous_owner_valid=self._previous_owner_valid,
            batch_size=self._batch_size,
            vectorized_fast_path=self._vectorized_delta_pack_profile,
        )
        timings["persistent_delta_pack_sec"] = time.perf_counter() - started

        self._ensure_layer_device()
        update_fn = self._update_fn_for_segments(int(delta_state["active"].shape[1]))
        defer_device_sync = (
            bool(request.device_only)
            and self._async_device_only_profile
            and not bool(request.synchronize_device)
        )
        started = time.perf_counter()
        delta_device = _copy_state_to_device(
            jax=self._jax,
            state={
                key: value
                for key, value in delta_state.items()
                if key not in {"next_owner_pos", "next_owner_valid"}
            },
            block_until_ready=not defer_device_sync,
        )
        compose_device = _copy_state_to_device(
            jax=self._jax,
            state=_persistent_compose_state(compact_state),
            block_until_ready=not defer_device_sync,
        )
        timings["host_to_device_sec"] = time.perf_counter() - started

        started = time.perf_counter()
        self._layer_device = update_fn(self._layer_device, delta_device)
        if not defer_device_sync:
            self._layer_device.block_until_ready()
        timings["persistent_update_sec"] = time.perf_counter() - started

        started = time.perf_counter()
        output_device = self._compose_fn(self._layer_device, compose_device)
        if not defer_device_sync:
            output_device.block_until_ready()
        self.last_output_device = output_device
        timings["device_render_sec"] = time.perf_counter() - started

        if bool(request.device_only):
            frames = out
            timings["device_to_host_sec"] = 0.0
        else:
            started = time.perf_counter()
            frames = np.asarray(output_device)
            timings["device_to_host_sec"] = time.perf_counter() - started
        timings["owner_ordered_pack_sec"] = 0.0
        timings["view_major_to_row_major_sec"] = 0.0
        if not bool(request.device_only):
            if out.shape[0] == self._batch_size * PLAYER_COUNT:
                out[...] = frames.reshape(
                    self._batch_size * PLAYER_COUNT,
                    1,
                    TARGET_SIZE,
                    TARGET_SIZE,
                )
            else:
                out[...] = frames[rows, players]
        self._previous_cursor = next_cursor
        self._previous_owner_pos = delta_state["next_owner_pos"]
        self._previous_owner_valid = delta_state["next_owner_valid"]
        self._previous_avatar_color = avatar_color.copy()
        partial_request = 0.0 if out.shape[0] == self._batch_size * PLAYER_COUNT else 1.0
        telemetry = {
            **timings,
            "render_sec": float(
                timings["production_to_compact_sec"]
                + timings["persistent_compact_state_handoff_sec"]
                + timings["persistent_delta_pack_sec"]
                + timings["host_to_device_sec"]
                + timings["persistent_update_sec"]
                + timings["device_render_sec"]
                + timings["device_to_host_sec"]
            ),
            "render_trail_slots": float(delta_stats["max_cursor"]),
            "active_trail_count_max": float(delta_stats["max_cursor"]),
            "render_truncation_row_count": 0.0,
            "partial_render_request": partial_request,
            "persistent_partial_render_request": partial_request,
            "render_output_count": float(rows.shape[0]),
            "persistent_reset_row_count": float(delta_stats["reset_row_count"]),
            "persistent_delta_slot_count": float(delta_stats["delta_slot_count"]),
            "persistent_delta_max_slots": float(delta_state["active"].shape[1]),
            "persistent_update_fn_cache_size": float(len(self._update_fn_cache)),
            "persistent_cache_initialized": 1.0,
            "persistent_profile_policy_space_direct64": 1.0,
            "persistent_async_device_only_profile": float(self._async_device_only_profile),
            "persistent_device_sync_deferred": float(defer_device_sync),
            "persistent_vectorized_delta_pack_profile": float(self._vectorized_delta_pack_profile),
            "persistent_compact_state_passthrough": compact_state_passthrough,
            "state_row_overlay_count": float(len(state_row_overlays)),
            "state_row_overlay_row_count": float(
                sum(np.asarray(overlay.rows).reshape(-1).shape[0] for overlay in state_row_overlays)
            ),
        }
        return SourceStateBatchedRenderResult(
            frames=out,
            telemetry=telemetry,
            device_frames=output_device,
        )

    def _validate_full_row_major_request(
        self,
        *,
        rows: Any,
        players: Any,
        out: Any,
    ) -> None:
        np = self._np
        expected_rows = _row_major_render_rows(np=np, batch_size=self._batch_size)
        expected_players = _row_major_render_players(np=np, batch_size=self._batch_size)
        is_full = rows.shape == expected_rows.shape and bool(np.array_equal(rows, expected_rows))
        if rows.shape != players.shape:
            raise ValueError(
                "controlled_players must have the same shape as row_indices; "
                f"got {players.shape} and {rows.shape}"
            )
        if rows.ndim != 1:
            raise ValueError(f"row_indices must be rank 1, got shape {rows.shape}")
        if bool((rows < 0).any()) or bool((rows >= self._batch_size).any()):
            raise ValueError(f"row_indices entries must be in [0, {self._batch_size})")
        if bool((players < 0).any()) or bool((players >= PLAYER_COUNT).any()):
            raise ValueError(f"controlled_players entries must be in [0, {PLAYER_COUNT})")
        if is_full and not bool(np.array_equal(players, expected_players)):
            raise ValueError(
                "full persistent render requests require row-major controlled players "
                "[0, 1] for every row"
            )
        expected_out_shape = (int(rows.shape[0]), 1, TARGET_SIZE, TARGET_SIZE)
        if out.shape != expected_out_shape:
            raise ValueError(
                "request.out must match persistent renderer output; "
                f"got {out.shape}, expected {expected_out_shape}"
            )
        if out.dtype != np.uint8:
            raise ValueError(f"request.out must be uint8, got {out.dtype}")

    def _ensure_layer_device(self) -> None:
        if self._layer_device is not None:
            return
        self._layer_device = self._jnp.full(
            (self._batch_size, PLAYER_COUNT, TARGET_SIZE, TARGET_SIZE),
            self._jnp.uint8(34),
            dtype=self._jnp.uint8,
        )

    def _update_fn_for_segments(self, max_segments: int) -> Any:
        slots = max(1, int(max_segments))
        cached = self._update_fn_cache.get(slots)
        if cached is not None:
            return cached
        cached = _make_persistent_policy_update_fn(
            jax=self._jax,
            jnp=self._jnp,
            target_size=self._target_size,
            map_size=self._map_size,
            max_segments=slots,
        )
        self._update_fn_cache[slots] = cached
        return cached


def _persistent_compact_state_from_production(
    *,
    np: Any,
    production_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if "visual_trail_pos" in production_state and "visual_trail_active" in production_state:
        return _persistent_visual_compact_state_from_production_fast(
            np=np,
            production_state=production_state,
            config=config,
        )
    compact_config = dict(config)
    compact_config["trail_slots"] = _persistent_live_trail_slots_from_production(
        np=np,
        production_state=production_state,
        config=config,
    )
    return _production_to_benchmark_source_state(
        np=np,
        production_state=production_state,
        config=compact_config,
    )


def _is_persistent_compact_render_state(state: Mapping[str, Any]) -> bool:
    return bool(state.get(PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_MARKER, False))


def _validate_persistent_compact_render_state(
    *,
    np: Any,
    state: Mapping[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    missing = sorted(
        key for key in PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_KEYS if key not in state
    )
    if missing:
        raise ValueError("persistent compact render state is missing keys: " + ", ".join(missing))
    batch_size = int(config["batch_size"])
    player_count = int(config["player_count"])
    bonus_count = int(config.get("bonus_count", 0))
    compact = {
        key: np.asarray(state[key]) for key in PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_KEYS
    }
    trail_shape = compact["trail_x"].shape
    if len(trail_shape) != 2 or trail_shape[0] != batch_size:
        raise ValueError(
            f"persistent compact trail arrays must have shape [batch, slots]; got {trail_shape}"
        )
    trail_slots = int(trail_shape[1])
    for key in ("trail_y", "trail_radius", "trail_owner", "trail_active", "trail_break_before"):
        if compact[key].shape != trail_shape:
            raise ValueError(
                f"{key} must match trail_x shape {trail_shape}, got {compact[key].shape}"
            )
    if compact["trail_write_cursor"].shape != (batch_size,):
        raise ValueError(
            "trail_write_cursor must have shape "
            f"{(batch_size,)}, got {compact['trail_write_cursor'].shape}"
        )
    player_shape = (batch_size, player_count)
    for key in ("head_x", "head_y", "head_radius", "head_alive", "avatar_color"):
        if compact[key].shape != player_shape:
            raise ValueError(f"{key} must have shape {player_shape}, got {compact[key].shape}")
    bonus_shape = (batch_size, bonus_count)
    for key in ("bonus_x", "bonus_y", "bonus_radius", "bonus_active", "bonus_type"):
        if compact[key].shape != bonus_shape:
            raise ValueError(f"{key} must have shape {bonus_shape}, got {compact[key].shape}")
    if trail_slots < 1:
        raise ValueError("persistent compact render state requires at least one trail slot")
    if not np.issubdtype(compact["trail_x"].dtype, np.floating):
        raise ValueError("trail_x must be floating point")
    if not np.issubdtype(compact["head_x"].dtype, np.floating):
        raise ValueError("head_x must be floating point")
    if not np.issubdtype(compact["trail_owner"].dtype, np.integer):
        raise ValueError("trail_owner must be integer")
    if not np.issubdtype(compact["avatar_color"].dtype, np.integer):
        raise ValueError("avatar_color must be integer")
    return compact


def _persistent_visual_compact_state_from_production_fast(
    *,
    np: Any,
    production_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Adapt reconstructed visual-trail state without rebuilding full trail arrays."""

    batch_size = int(config["batch_size"])
    player_count = int(config["player_count"])
    bonus_count = int(config["bonus_count"])
    geometry_dtype = (
        np.float64
        if str(config.get("geometry_dtype", GEOMETRY_DTYPE_FLOAT32)) == GEOMETRY_DTYPE_FLOAT64
        else np.float32
    )
    trail_slots = _persistent_live_trail_slots_from_production(
        np=np,
        production_state=production_state,
        config=config,
    )

    trail_pos = np.asarray(production_state["visual_trail_pos"])[
        :batch_size,
        :trail_slots,
    ]
    if trail_pos.dtype != geometry_dtype:
        trail_pos = trail_pos.astype(geometry_dtype, copy=False)
    trail_radius = np.asarray(production_state["visual_trail_radius"])[
        :batch_size,
        :trail_slots,
    ]
    if trail_radius.dtype != geometry_dtype:
        trail_radius = trail_radius.astype(geometry_dtype, copy=False)
    trail_owner = np.asarray(
        production_state["visual_trail_owner"],
        dtype=np.int32,
    )[:batch_size, :trail_slots]
    trail_write_cursor = np.clip(
        np.asarray(
            production_state.get(
                "visual_trail_write_cursor",
                np.full((batch_size,), trail_slots, dtype=np.int32),
            ),
            dtype=np.int32,
        )[:batch_size],
        0,
        trail_slots,
    ).astype(np.int32, copy=False)
    trail_active = np.asarray(
        production_state["visual_trail_active"],
        dtype=np.uint8,
    )[:batch_size, :trail_slots].copy()
    for row in range(batch_size):
        cursor = int(trail_write_cursor[row])
        if cursor < trail_slots:
            trail_active[row, max(0, cursor) :] = 0
    trail_break = np.asarray(
        production_state.get(
            "visual_trail_break_before",
            np.zeros(trail_active.shape, dtype=bool),
        ),
        dtype=np.uint8,
    )[:batch_size, :trail_slots]

    pos = np.asarray(production_state["pos"])[:batch_size, :player_count]
    if pos.dtype != geometry_dtype:
        pos = pos.astype(geometry_dtype, copy=False)
    radius = np.asarray(production_state["radius"])[:batch_size, :player_count]
    if radius.dtype != geometry_dtype:
        radius = radius.astype(geometry_dtype, copy=False)
    alive = np.asarray(production_state["alive"], dtype=np.uint8)[
        :batch_size,
        :player_count,
    ]
    present = np.asarray(
        production_state.get("present", production_state["alive"]),
        dtype=np.uint8,
    )[:batch_size, :player_count]
    avatar_color = np.asarray(
        production_state.get(
            "avatar_color",
            np.arange(player_count, dtype=np.int16)[None, :].repeat(batch_size, axis=0),
        ),
        dtype=np.int32,
    )[:batch_size, :player_count]

    if bonus_count and "bonus_active" in production_state:
        bonus_pos = np.asarray(production_state["bonus_pos"])[:batch_size, :bonus_count]
        if bonus_pos.dtype != geometry_dtype:
            bonus_pos = bonus_pos.astype(geometry_dtype, copy=False)
        bonus_radius = np.asarray(production_state["bonus_radius"])[
            :batch_size,
            :bonus_count,
        ]
        if bonus_radius.dtype != geometry_dtype:
            bonus_radius = bonus_radius.astype(geometry_dtype, copy=False)
        bonus_active = np.asarray(production_state["bonus_active"], dtype=np.uint8)[
            :batch_size,
            :bonus_count,
        ]
        bonus_type = np.asarray(production_state["bonus_type"], dtype=np.int32)[
            :batch_size,
            :bonus_count,
        ]
    else:
        bonus_pos = np.zeros((batch_size, bonus_count, 2), dtype=geometry_dtype)
        bonus_radius = np.zeros((batch_size, bonus_count), dtype=geometry_dtype)
        bonus_active = np.zeros((batch_size, bonus_count), dtype=np.uint8)
        bonus_type = np.ones((batch_size, bonus_count), dtype=np.int32)

    return {
        "trail_x": trail_pos[..., 0],
        "trail_y": trail_pos[..., 1],
        "trail_radius": trail_radius,
        "trail_owner": trail_owner,
        "trail_active": trail_active,
        "trail_break_before": trail_break,
        "head_x": pos[..., 0],
        "head_y": pos[..., 1],
        "head_radius": radius,
        "head_alive": (alive & present).astype(np.uint8, copy=False),
        "avatar_color": avatar_color,
        "trail_write_cursor": trail_write_cursor,
        "bonus_x": bonus_pos[..., 0] if bonus_count else bonus_pos.reshape(batch_size, 0),
        "bonus_y": bonus_pos[..., 1] if bonus_count else bonus_pos.reshape(batch_size, 0),
        "bonus_radius": bonus_radius,
        "bonus_active": bonus_active,
        "bonus_type": bonus_type,
    }


def _persistent_live_trail_slots_from_production(
    *,
    np: Any,
    production_state: dict[str, Any],
    config: dict[str, Any],
) -> int:
    max_slots = int(config.get("body_capacity", config["trail_slots"]))
    if max_slots < 1:
        return 1
    if "visual_trail_active" in production_state:
        active = np.asarray(production_state["visual_trail_active"])
        cursor = np.asarray(
            production_state.get(
                "visual_trail_write_cursor",
                np.full((int(config["batch_size"]),), active.shape[1], dtype=np.int32),
            ),
            dtype=np.int32,
        )
    else:
        active = np.asarray(production_state["body_active"])
        cursor = np.asarray(
            production_state.get(
                "body_write_cursor",
                np.full((int(config["batch_size"]),), active.shape[1], dtype=np.int32),
            ),
            dtype=np.int32,
        )
    capacity = min(max_slots, int(active.shape[1]))
    live_slots = int(np.clip(cursor, 0, capacity).max()) if cursor.size else 0
    return max(1, min(capacity, live_slots))


def _persistent_delta_state(
    *,
    np: Any,
    compact_state: dict[str, Any],
    previous_cursor: Any | None,
    previous_owner_pos: Any | None,
    previous_owner_valid: Any | None,
    batch_size: int,
    vectorized_fast_path: bool = False,
) -> tuple[dict[str, Any], Any, dict[str, int]]:
    cursor = np.asarray(compact_state["trail_write_cursor"], dtype=np.int32)[: int(batch_size)]
    capacity = int(np.asarray(compact_state["trail_active"]).shape[1])
    cursor = np.clip(cursor, 0, capacity).astype(np.int32, copy=True)
    if previous_cursor is None:
        previous = np.zeros_like(cursor)
        reset_mask = np.ones_like(cursor, dtype=np.uint8)
    else:
        previous = np.asarray(previous_cursor, dtype=np.int32)
        if previous.shape != cursor.shape:
            previous = np.zeros_like(cursor)
            reset_mask = np.ones_like(cursor, dtype=np.uint8)
        else:
            reset_mask = (cursor < previous).astype(np.uint8)
            previous = np.where(reset_mask != 0, 0, previous).astype(np.int32, copy=False)
    player_count = PLAYER_COUNT
    owner_pos = (
        np.zeros((int(batch_size), player_count, 2), dtype=np.float64)
        if previous_owner_pos is None
        else np.asarray(previous_owner_pos, dtype=np.float64).copy()
    )
    owner_valid = (
        np.zeros((int(batch_size), player_count), dtype=bool)
        if previous_owner_valid is None
        else np.asarray(previous_owner_valid, dtype=bool).copy()
    )
    if owner_pos.shape != (int(batch_size), player_count, 2) or owner_valid.shape != (
        int(batch_size),
        player_count,
    ):
        owner_pos = np.zeros((int(batch_size), player_count, 2), dtype=np.float64)
        owner_valid = np.zeros((int(batch_size), player_count), dtype=bool)
        reset_mask = np.ones_like(cursor, dtype=np.uint8)
        previous = np.zeros_like(cursor)
    reset_rows = np.flatnonzero(reset_mask)
    if reset_rows.size:
        owner_pos[reset_rows] = 0.0
        owner_valid[reset_rows] = False
    geometry_dtype = np.asarray(compact_state["trail_x"]).dtype
    trail_active = np.asarray(compact_state["trail_active"], dtype=np.uint8)
    start = np.minimum(previous, cursor)
    span_counts = np.maximum(cursor - start, 0).astype(np.int32, copy=False)
    max_delta = max(1, int(span_counts.max()) if span_counts.size else 1)
    x0 = np.zeros((int(batch_size), max_delta), dtype=geometry_dtype)
    y0 = np.zeros_like(x0)
    x1 = np.zeros_like(x0)
    y1 = np.zeros_like(x0)
    radius = np.zeros_like(x0)
    owner = np.full((int(batch_size), max_delta), -1, dtype=np.int32)
    active = np.zeros((int(batch_size), max_delta), dtype=np.uint8)
    trail_owner = np.asarray(compact_state["trail_owner"], dtype=np.int32)
    trail_x = np.asarray(compact_state["trail_x"], dtype=geometry_dtype)
    trail_y = np.asarray(compact_state["trail_y"], dtype=geometry_dtype)
    trail_radius = np.asarray(compact_state["trail_radius"], dtype=geometry_dtype)
    trail_break = np.asarray(compact_state["trail_break_before"], dtype=np.uint8)
    avatar_color = np.asarray(compact_state["avatar_color"], dtype=np.int32)[: int(batch_size)]
    offsets = np.arange(max_delta, dtype=np.int32)
    row_index = np.arange(int(batch_size), dtype=np.int32)[:, None]
    valid_span = offsets[None, :] < span_counts[:, None]
    slot_index = start[:, None] + offsets[None, :]
    safe_slot_index = np.where(valid_span, slot_index, 0)
    span_active = trail_active[row_index, safe_slot_index] != 0
    span_owner = trail_owner[row_index, safe_slot_index]
    span_owner_valid = (span_owner >= 0) & (span_owner < player_count)
    vectorized_span = bool(vectorized_fast_path) and bool(
        np.logical_or(~valid_span, span_active & span_owner_valid).all()
    )
    if vectorized_span:
        delta_slot_count = int(span_counts.sum())
        for out_slot in range(max_delta):
            rows = np.flatnonzero(valid_span[:, out_slot])
            if not rows.size:
                continue
            slots = safe_slot_index[rows, out_slot]
            slot_owner = span_owner[rows, out_slot].astype(np.intp, copy=False)
            current_x = trail_x[rows, slots]
            current_y = trail_y[rows, slots]
            connects = owner_valid[rows, slot_owner] & (trail_break[rows, slots] == 0)
            previous_x = owner_pos[rows, slot_owner, 0].astype(geometry_dtype, copy=False)
            previous_y = owner_pos[rows, slot_owner, 1].astype(geometry_dtype, copy=False)
            x0[rows, out_slot] = np.where(connects, previous_x, current_x)
            y0[rows, out_slot] = np.where(connects, previous_y, current_y)
            x1[rows, out_slot] = current_x
            y1[rows, out_slot] = current_y
            radius[rows, out_slot] = trail_radius[rows, slots]
            owner[rows, out_slot] = slot_owner
            active[rows, out_slot] = 1
            owner_pos[rows, slot_owner, 0] = current_x
            owner_pos[rows, slot_owner, 1] = current_y
            owner_valid[rows, slot_owner] = True
    else:
        counts = np.asarray(
            [
                int(np.count_nonzero(trail_active[row, int(start[row]) : int(cursor[row])]))
                for row in range(int(batch_size))
            ],
            dtype=np.int32,
        )
        compact_max_delta = max(1, int(counts.max()) if counts.size else 1)
        if compact_max_delta != max_delta:
            max_delta = compact_max_delta
            x0 = np.zeros((int(batch_size), max_delta), dtype=geometry_dtype)
            y0 = np.zeros_like(x0)
            x1 = np.zeros_like(x0)
            y1 = np.zeros_like(x0)
            radius = np.zeros_like(x0)
            owner = np.full((int(batch_size), max_delta), -1, dtype=np.int32)
            active = np.zeros((int(batch_size), max_delta), dtype=np.uint8)
        delta_slot_count = 0
        for row in range(int(batch_size)):
            out_slot = 0
            row_start = int(start[row])
            row_cursor = int(cursor[row])
            for slot in range(row_start, row_cursor):
                if not bool(trail_active[row, slot]):
                    continue
                slot_owner = int(trail_owner[row, slot])
                if slot_owner < 0 or slot_owner >= player_count:
                    continue
                current_pos = (float(trail_x[row, slot]), float(trail_y[row, slot]))
                if bool(owner_valid[row, slot_owner]) and not bool(trail_break[row, slot]):
                    seg_start = (
                        float(owner_pos[row, slot_owner, 0]),
                        float(owner_pos[row, slot_owner, 1]),
                    )
                else:
                    seg_start = current_pos
                x0[row, out_slot] = seg_start[0]
                y0[row, out_slot] = seg_start[1]
                x1[row, out_slot] = current_pos[0]
                y1[row, out_slot] = current_pos[1]
                radius[row, out_slot] = trail_radius[row, slot]
                owner[row, out_slot] = slot_owner
                active[row, out_slot] = 1
                out_slot += 1
                delta_slot_count += 1
                owner_pos[row, slot_owner] = current_pos
                owner_valid[row, slot_owner] = True
    return (
        {
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "radius": radius,
            "owner": owner,
            "active": active,
            "reset_mask": reset_mask,
            "avatar_color": avatar_color,
            "next_owner_pos": owner_pos,
            "next_owner_valid": owner_valid,
        },
        cursor,
        {
            "reset_row_count": int(np.count_nonzero(reset_mask)),
            "delta_slot_count": int(delta_slot_count),
            "max_cursor": int(cursor.max()) if cursor.size else 0,
        },
    )


def _persistent_compose_state(compact_state: dict[str, Any]) -> dict[str, Any]:
    dtype = compact_state["head_x"].dtype
    return {
        "trail_x": compact_state["head_x"][:, :1].astype(dtype, copy=True) * 0,
        "head_x": compact_state["head_x"],
        "head_y": compact_state["head_y"],
        "head_radius": compact_state["head_radius"],
        "head_alive": compact_state["head_alive"],
        "avatar_color": compact_state["avatar_color"],
        "bonus_x": compact_state["bonus_x"],
        "bonus_y": compact_state["bonus_y"],
        "bonus_radius": compact_state["bonus_radius"],
        "bonus_active": compact_state["bonus_active"],
        "bonus_type": compact_state["bonus_type"],
    }


def _make_persistent_policy_update_fn(
    *,
    jax: Any,
    jnp: Any,
    target_size: int,
    map_size: float,
    max_segments: int,
) -> Any:
    grid_x = (
        (jnp.arange(int(target_size), dtype=jnp.float32) + 0.5)
        * float(map_size)
        / float(target_size)
    )
    grid_y = (
        (jnp.arange(int(target_size), dtype=jnp.float32) + 0.5)
        * float(map_size)
        / float(target_size)
    )
    world_x = grid_x[None, :]
    world_y = grid_y[:, None]
    background = jnp.uint8(34)

    @jax.jit
    def update(layer: Any, delta: dict[str, Any]) -> Any:
        reset = delta["reset_mask"].astype(bool)
        current = jnp.where(reset[:, None, None, None], background, layer)

        def draw_slot(slot: Any, frame: Any) -> Any:
            active = (jnp.take(delta["active"], slot, axis=1) != 0) & (
                jnp.take(delta["owner"], slot, axis=1) >= 0
            )
            x0 = jnp.take(delta["x0"], slot, axis=1)[:, None, None]
            y0 = jnp.take(delta["y0"], slot, axis=1)[:, None, None]
            x1 = jnp.take(delta["x1"], slot, axis=1)[:, None, None]
            y1 = jnp.take(delta["y1"], slot, axis=1)[:, None, None]
            radius = jnp.take(delta["radius"], slot, axis=1)[:, None, None]
            vx = x1 - x0
            vy = y1 - y0
            length_sq = jnp.maximum(vx * vx + vy * vy, 1.0e-4)
            t = jnp.clip(((world_x - x0) * vx + (world_y - y0) * vy) / length_sq, 0.0, 1.0)
            nearest_x = x0 + t * vx
            nearest_y = y0 + t * vy
            dx = world_x - nearest_x
            dy = world_y - nearest_y
            hit = (dx * dx + dy * dy <= radius * radius) & active[:, None, None]
            owner = jnp.clip(jnp.take(delta["owner"], slot, axis=1), 0, PLAYER_COUNT - 1)
            avatar_color = delta["avatar_color"][:, :PLAYER_COUNT].astype(jnp.int32)
            owner_color = jnp.take_along_axis(avatar_color, owner[:, None], axis=1)[:, 0]
            view0_value = jnp.where(
                owner_color == avatar_color[:, 0],
                jnp.uint8(96),
                jnp.uint8(128),
            )
            view1_value = jnp.where(
                owner_color == avatar_color[:, 1],
                jnp.uint8(96),
                jnp.uint8(128),
            )
            values = jnp.stack((view0_value, view1_value), axis=1)
            painted = jnp.where(hit[:, None, :, :], values[:, :, None, None], frame)
            return jnp.maximum(frame, painted)

        return jax.lax.fori_loop(0, int(max_segments), draw_slot, current)

    return update


def _make_persistent_policy_compose_fn(
    *,
    jax: Any,
    jnp: Any,
    target_size: int,
    map_size: float,
    bonus_count: int,
    bonus_render_mode_id: int,
) -> Any:
    grid_x = (
        (jnp.arange(int(target_size), dtype=jnp.float32) + 0.5)
        * float(map_size)
        / float(target_size)
    )
    grid_y = (
        (jnp.arange(int(target_size), dtype=jnp.float32) + 0.5)
        * float(map_size)
        / float(target_size)
    )
    world_x = grid_x[None, None, :]
    world_y = grid_y[None, :, None]

    @jax.jit
    def compose(layer: Any, state: dict[str, Any]) -> Any:
        views = []
        for controlled_player in range(PLAYER_COUNT):
            out = layer[:, controlled_player, :, :]
            out = _draw_direct_bonus_and_heads(
                jnp=jnp,
                lax=jax.lax,
                out=out,
                state=state,
                world_x=world_x,
                world_y=world_y,
                target_size=int(target_size),
                map_size=float(map_size),
                bonus_count=int(bonus_count),
                bonus_render_mode_id=int(bonus_render_mode_id),
                controlled_player=controlled_player,
                player_luma_by_index=(96, 128),
            )
            views.append(out)
        return jnp.stack(views, axis=1)[:, :, None, :, :].astype(jnp.uint8)

    return compose


def _make_profile_observation_renderer(
    *,
    jax: Any,
    jnp: Any,
    np: Any,
    checked: dict[str, Any],
    render_fn_for_slots: Any,
    bonus_render_mode_id: int,
) -> Any:
    backend = str(checked["observation_renderer_backend"])
    if backend == SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND:
        return _DynamicJaxBatchedObservationRenderer(
            jax=jax,
            np=np,
            config=checked["render_config"],
            render_fn_for_slots=render_fn_for_slots,
        )
    if backend == SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND:
        return _PersistentJaxPolicyFramebufferRenderer(
            jax=jax,
            jnp=jnp,
            np=np,
            config=checked["render_config"],
            bonus_render_mode_id=bonus_render_mode_id,
        )
    raise ValueError(f"unsupported observation_renderer_backend {backend!r}")


class _DynamicJaxBatchedObservationRenderer:
    backend_name = "jax_gpu_batched_profile"

    def __init__(
        self,
        *,
        jax: Any,
        np: Any,
        config: dict[str, Any],
        render_fn_for_slots: Any,
    ) -> None:
        self._jax = jax
        self._np = np
        self._config = config
        self._render_fn_for_slots = render_fn_for_slots

    def render(self, request: SourceStateBatchedRenderRequest) -> SourceStateBatchedRenderResult:
        np = self._np
        batch_size = int(self._config["batch_size"])
        expected_rows = _row_major_render_rows(np=np, batch_size=batch_size)
        expected_players = _row_major_render_players(np=np, batch_size=batch_size)
        rows = np.asarray(request.row_indices, dtype=np.int64)
        players = np.asarray(request.controlled_players, dtype=np.int64)
        if rows.shape != players.shape:
            raise ValueError(
                "controlled_players must have the same shape as row_indices; "
                f"got {players.shape} and {rows.shape}"
            )
        if rows.ndim != 1:
            raise ValueError(f"row_indices must be rank 1, got shape {rows.shape}")
        if bool((rows < 0).any()) or bool((rows >= batch_size).any()):
            raise ValueError(f"row_indices entries must be in [0, {batch_size})")
        if bool((players < 0).any()) or bool((players >= PLAYER_COUNT).any()):
            raise ValueError(f"controlled_players entries must be in [0, {PLAYER_COUNT})")
        out = np.asarray(request.out)
        expected_out_shape = (int(rows.shape[0]), 1, TARGET_SIZE, TARGET_SIZE)
        if out.shape != expected_out_shape:
            raise ValueError(
                "request.out must match requested renderer output; "
                f"got {out.shape}, expected {expected_out_shape}"
            )
        if out.dtype != np.uint8:
            raise ValueError(f"request.out must be uint8, got {out.dtype}")
        is_full_row_major = (
            rows.shape == expected_rows.shape
            and bool(np.array_equal(rows, expected_rows))
            and bool(np.array_equal(players, expected_players))
        )

        production_state = source_state_render_state_with_row_overlays(
            request.state,
            request.state_row_overlays,
            batch_size=batch_size,
        )
        frames, timings, trail_stats = _render_candidate_frames_from_production_state(
            jax=self._jax,
            np=np,
            production_state=production_state,
            config=self._config,
            render_fn_for_slots=self._render_fn_for_slots,
        )
        flat_frames = frames.reshape(batch_size * PLAYER_COUNT, 1, TARGET_SIZE, TARGET_SIZE)
        if is_full_row_major:
            out[...] = flat_frames
        else:
            out[...] = frames[rows, players]
        telemetry = {
            **timings,
            "render_sec": _candidate_render_total_sec(timings),
            "render_trail_slots": float(trail_stats["render_trail_slots"]),
            "active_trail_count_max": float(trail_stats["active_trail_count_max"]),
            "render_truncation_row_count": float(trail_stats["render_truncation_row_count"]),
            "partial_render_request": 0.0 if is_full_row_major else 1.0,
            "render_output_count": float(rows.shape[0]),
            "state_row_overlay_count": float(len(tuple(request.state_row_overlays))),
        }
        return SourceStateBatchedRenderResult(
            frames=out,
            telemetry=telemetry,
        )


class _JaxHybridPolicySearchProbe:
    """Profile-only batched model/search pressure stand-in.

    This is not LightZero MCTS. It exists to measure whether batched observation
    rows can feed repeated GPU model-like work without immediately losing the
    batch to Python scalar loops.
    """

    backend_name = "jax_policy_search_pressure_probe"
    semantics = "synthetic_jax_conv_policy_pressure"

    def __init__(
        self,
        *,
        jax: Any,
        jnp: Any,
        simulations: int,
        channels: int,
    ) -> None:
        self._jax = jax
        self._jnp = jnp
        self._simulations = _positive_int(simulations, "hybrid_policy_probe_simulations")
        self._channels = _positive_int(channels, "hybrid_policy_probe_channels")
        scale = 1.0 / float(3 * 3 * POLICY_FRAME_STACK_DEPTH)
        self._kernel = jnp.full(
            (3, 3, POLICY_FRAME_STACK_DEPTH, self._channels),
            scale,
            dtype=jnp.float32,
        )
        self._policy = jnp.linspace(
            0.01,
            0.03,
            self._channels * ACTION_COUNT,
            dtype=jnp.float32,
        ).reshape(self._channels, ACTION_COUNT)
        self._probe_fn = jax.jit(self._run_device)

    def run(self, flat_obs: Any) -> HybridPolicySearchProbeResult:
        jax = self._jax
        jnp = self._jnp
        root_count = int(getattr(flat_obs, "shape", [0])[0])
        started = time.perf_counter()
        obs = jax.device_put(jnp.asarray(flat_obs, dtype=jnp.float32))
        obs.block_until_ready()
        host_to_device_sec = time.perf_counter() - started

        started = time.perf_counter()
        output = self._probe_fn(obs)
        output.block_until_ready()
        device_sec = time.perf_counter() - started

        started = time.perf_counter()
        _ = float(output)
        readback_sec = time.perf_counter() - started
        return HybridPolicySearchProbeResult(
            telemetry={
                "host_to_device_sec": host_to_device_sec,
                "device_sec": device_sec,
                "readback_sec": readback_sec,
                "total_sec": host_to_device_sec + device_sec + readback_sec,
                "simulations": float(self._simulations),
                "channels": float(self._channels),
                "roots": float(root_count),
                "model_eval_count": float(root_count * self._simulations),
                "output_readback_bytes": 8.0,
                "action_mask_consumed": 0.0,
                "compile_excluded_by_warmup": 1.0,
            }
        )

    def _run_device(self, obs: Any) -> Any:
        jnp = self._jnp
        x = jnp.transpose(obs, (0, 2, 3, 1))
        acc = jnp.asarray(0.0, dtype=jnp.float32)
        for _ in range(self._simulations):
            y = self._jax.lax.conv_general_dilated(
                x,
                self._kernel,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            y = jnp.tanh(y)
            pooled = jnp.mean(y, axis=(1, 2))
            logits = pooled @ self._policy
            acc = acc + jnp.sum(logits)
            x = jnp.repeat(jnp.mean(y, axis=-1, keepdims=True), POLICY_FRAME_STACK_DEPTH, axis=-1)
        return acc


class _JaxHybridBatchedStackProbe:
    """Profile-only consumer for the batched stack before CPU scalarization."""

    backend_name = "jax_batched_stack_policy_pressure_probe"
    semantics = "synthetic_jax_batched_stack_device_handoff_pressure"

    def __init__(
        self,
        *,
        jax: Any,
        jnp: Any,
        simulations: int,
        channels: int,
        device_latest_provider: Any | None = None,
    ) -> None:
        self._jax = jax
        self._jnp = jnp
        self._simulations = _positive_int(simulations, "hybrid_batched_stack_probe_simulations")
        self._channels = _positive_int(channels, "hybrid_batched_stack_probe_channels")
        self._device_latest_provider = device_latest_provider
        self._device_stack: Any | None = None
        scale = 1.0 / float(3 * 3 * POLICY_FRAME_STACK_DEPTH)
        self._kernel = jnp.full(
            (3, 3, POLICY_FRAME_STACK_DEPTH, self._channels),
            scale,
            dtype=jnp.float32,
        )
        self._policy = jnp.linspace(
            0.01,
            0.03,
            self._channels * ACTION_COUNT,
            dtype=jnp.float32,
        ).reshape(self._channels, ACTION_COUNT)
        self._normalize_fn = jax.jit(self._normalize_device)
        self._probe_fn = jax.jit(self._run_device)

    def run(self, observation: Any, action_mask: Any) -> HybridBatchedStackProbeResult:
        jax = self._jax
        jnp = self._jnp
        stack_shape = tuple(int(dim) for dim in getattr(observation, "shape", ()))
        if len(stack_shape) != 5:
            raise ValueError(
                f"batched stack probe expects observation shape [B,P,4,64,64], got {stack_shape}"
            )
        root_count = int(stack_shape[0] * stack_shape[1])
        latest_device = (
            getattr(self._device_latest_provider, "last_output_device", None)
            if self._device_latest_provider is not None
            else None
        )
        started = time.perf_counter()
        mask = jax.device_put(jnp.asarray(action_mask, dtype=jnp.bool_))
        mask.block_until_ready()
        host_to_device_sec = time.perf_counter() - started

        device_stack_update_sec = 0.0
        device_latest_source = 0.0
        if latest_device is None:
            started = time.perf_counter()
            stack = jax.device_put(jnp.asarray(observation))
            stack.block_until_ready()
            host_to_device_sec += time.perf_counter() - started
            host_to_device_bytes = float(getattr(observation, "nbytes", 0)) + float(
                getattr(action_mask, "nbytes", 0)
            )
        else:
            expected_latest_shape = (
                stack_shape[0],
                stack_shape[1],
                1,
                TARGET_SIZE,
                TARGET_SIZE,
            )
            latest_shape = tuple(int(dim) for dim in getattr(latest_device, "shape", ()))
            if latest_shape != expected_latest_shape:
                raise ValueError(
                    "device latest frame shape mismatch; "
                    f"got {latest_shape}, expected {expected_latest_shape}"
                )
            started = time.perf_counter()
            if (
                self._device_stack is None
                or tuple(int(dim) for dim in getattr(self._device_stack, "shape", ()))
                != stack_shape
            ):
                self._device_stack = jnp.zeros(stack_shape, dtype=latest_device.dtype)
            self._device_stack = jnp.concatenate(
                (self._device_stack[:, :, 1:], latest_device),
                axis=2,
            )
            self._device_stack.block_until_ready()
            device_stack_update_sec = time.perf_counter() - started
            stack = self._device_stack
            device_latest_source = 1.0
            host_to_device_bytes = float(getattr(action_mask, "nbytes", 0))

        started = time.perf_counter()
        normalized = self._normalize_fn(stack)
        normalized.block_until_ready()
        normalize_sec = time.perf_counter() - started

        started = time.perf_counter()
        output = self._probe_fn(normalized, mask)
        output.block_until_ready()
        device_sec = time.perf_counter() - started

        started = time.perf_counter()
        _ = float(output)
        readback_sec = time.perf_counter() - started
        return HybridBatchedStackProbeResult(
            telemetry={
                "host_to_device_sec": host_to_device_sec,
                "host_to_device_bytes": host_to_device_bytes,
                "device_latest_source": device_latest_source,
                "device_stack_update_sec": device_stack_update_sec,
                "normalize_sec": normalize_sec,
                "device_sec": device_sec,
                "readback_sec": readback_sec,
                "total_sec": host_to_device_sec + normalize_sec + device_sec + readback_sec,
                "simulations": float(self._simulations),
                "channels": float(self._channels),
                "roots": float(root_count),
                "input_rank": float(len(stack_shape)),
                "input_bytes": float(getattr(observation, "nbytes", 0)),
                "model_eval_count": float(root_count * self._simulations),
                "output_readback_bytes": 8.0,
                "action_mask_consumed": 1.0,
                "compile_excluded_by_warmup": 1.0,
            }
        )

    def _normalize_device(self, stack: Any) -> Any:
        jnp = self._jnp
        x = stack.astype(jnp.float32)
        if stack.dtype == jnp.uint8:
            x = x * jnp.asarray(1.0 / 255.0, dtype=jnp.float32)
        return x.reshape((-1, POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE))

    def _run_device(self, flat_stack: Any, action_mask: Any) -> Any:
        jnp = self._jnp
        x = jnp.transpose(flat_stack, (0, 2, 3, 1))
        mask = action_mask.reshape((-1, ACTION_COUNT)).astype(jnp.float32)
        acc = jnp.asarray(0.0, dtype=jnp.float32)
        for _ in range(self._simulations):
            y = self._jax.lax.conv_general_dilated(
                x,
                self._kernel,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            y = jnp.tanh(y)
            pooled = jnp.mean(y, axis=(1, 2))
            logits = pooled @ self._policy
            acc = acc + jnp.sum(logits * mask)
            x = jnp.repeat(jnp.mean(y, axis=-1, keepdims=True), POLICY_FRAME_STACK_DEPTH, axis=-1)
        return acc


class _JaxHybridResidentChunkProbe:
    """Profile-only resident replay/search pressure probe.

    This is not LightZero MCTS or a trainer. It keeps a compact uint8 replay-like
    buffer on device, writes the current batched roots into it, samples a fixed
    batch, runs synthetic policy/search pressure on current roots, and runs a
    tiny replay-train-shaped pass over sampled rows. It exists to test whether
    the batch can stay useful before scalar LightZero timestep materialization.
    """

    backend_name = "jax_resident_chunk_replay_search_probe"
    semantics = "synthetic_jax_resident_replay_search_pressure"

    def __init__(
        self,
        *,
        jax: Any,
        jnp: Any,
        simulations: int,
        channels: int,
        replay_steps: int,
        sample_batch_size: int,
        replay_train_steps: int,
        readback_checksum: bool,
    ) -> None:
        self._jax = jax
        self._jnp = jnp
        self._simulations = _positive_int(simulations, "hybrid_batched_stack_probe_simulations")
        self._channels = _positive_int(channels, "hybrid_batched_stack_probe_channels")
        self._replay_steps = _positive_int(replay_steps, "hybrid_resident_replay_steps")
        self._sample_batch_size = _positive_int(
            sample_batch_size,
            "hybrid_resident_sample_batch_size",
        )
        self._replay_train_steps = _positive_int(
            replay_train_steps,
            "hybrid_resident_replay_train_steps",
        )
        self._readback_checksum = bool(readback_checksum)
        scale = 1.0 / float(3 * 3 * POLICY_FRAME_STACK_DEPTH)
        self._kernel = jnp.full(
            (3, 3, POLICY_FRAME_STACK_DEPTH, self._channels),
            scale,
            dtype=jnp.float32,
        )
        self._policy = jnp.linspace(
            0.01,
            0.03,
            self._channels * ACTION_COUNT,
            dtype=jnp.float32,
        ).reshape(self._channels, ACTION_COUNT)
        self._replay_obs: Any | None = None
        self._replay_mask: Any | None = None
        self._cursor: Any | None = None
        self._valid_count: Any | None = None
        self._cursor_host = 0
        self._valid_count_host = 0
        self._capacity = 0
        self._root_count = 0
        self._write_fn = jax.jit(self._write_device)
        self._sample_fn = jax.jit(self._sample_device)
        self._policy_search_fn = jax.jit(self._policy_search_device)
        self._replay_train_fn = jax.jit(self._replay_train_device)

    def run(self, observation: Any, action_mask: Any) -> HybridBatchedStackProbeResult:
        jax = self._jax
        jnp = self._jnp
        stack_shape = tuple(int(dim) for dim in getattr(observation, "shape", ()))
        if len(stack_shape) != 5:
            raise ValueError(
                f"resident chunk probe expects observation shape [B,P,4,64,64], got {stack_shape}"
            )
        if stack_shape[2:] != (POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE):
            raise ValueError(
                "resident chunk probe expects stack tail "
                f"({POLICY_FRAME_STACK_DEPTH}, {TARGET_SIZE}, {TARGET_SIZE}), "
                f"got {stack_shape[2:]}"
            )
        mask_shape = tuple(int(dim) for dim in getattr(action_mask, "shape", ()))
        expected_mask_shape = stack_shape[:2] + (ACTION_COUNT,)
        if mask_shape != expected_mask_shape:
            raise ValueError(
                "resident chunk probe action_mask shape mismatch; "
                f"got {mask_shape}, expected {expected_mask_shape}"
            )
        root_count = int(stack_shape[0] * stack_shape[1])
        self._ensure_replay(root_count=root_count)

        started = time.perf_counter()
        stack = jax.device_put(jnp.asarray(observation, dtype=jnp.uint8))
        mask = jax.device_put(jnp.asarray(action_mask, dtype=jnp.bool_))
        stack.block_until_ready()
        mask.block_until_ready()
        host_to_device_sec = time.perf_counter() - started
        host_to_device_bytes = float(getattr(observation, "nbytes", 0)) + float(
            getattr(action_mask, "nbytes", 0)
        )

        flat_stack = stack.reshape((-1, POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE))
        flat_mask = mask.reshape((-1, ACTION_COUNT))

        started = time.perf_counter()
        (
            self._replay_obs,
            self._replay_mask,
            self._cursor,
            self._valid_count,
            write_checksum,
        ) = self._write_fn(
            self._replay_obs,
            self._replay_mask,
            self._cursor,
            self._valid_count,
            flat_stack,
            flat_mask,
        )
        write_checksum.block_until_ready()
        replay_write_sec = time.perf_counter() - started
        self._cursor_host = (self._cursor_host + root_count) % int(self._capacity)
        self._valid_count_host = min(self._valid_count_host + root_count, int(self._capacity))

        started = time.perf_counter()
        sample_obs, sample_mask, sample_checksum = self._sample_fn(
            self._replay_obs,
            self._replay_mask,
            self._cursor,
            self._valid_count,
        )
        sample_checksum.block_until_ready()
        replay_sample_sec = time.perf_counter() - started

        started = time.perf_counter()
        policy_checksum = self._policy_search_fn(flat_stack, flat_mask)
        policy_checksum.block_until_ready()
        policy_search_sec = time.perf_counter() - started

        started = time.perf_counter()
        train_checksum = self._replay_train_fn(sample_obs, sample_mask)
        train_checksum.block_until_ready()
        replay_train_sec = time.perf_counter() - started

        started = time.perf_counter()
        checksum = write_checksum + sample_checksum + policy_checksum + train_checksum
        checksum_value = float(checksum) if self._readback_checksum else 0.0
        readback_sec = time.perf_counter() - started
        total_sec = (
            host_to_device_sec
            + replay_write_sec
            + replay_sample_sec
            + policy_search_sec
            + replay_train_sec
            + readback_sec
        )

        return HybridBatchedStackProbeResult(
            telemetry={
                "host_to_device_sec": host_to_device_sec,
                "host_to_device_bytes": host_to_device_bytes,
                "device_latest_source": 0.0,
                "device_stack_update_sec": 0.0,
                "normalize_sec": 0.0,
                "device_sec": replay_write_sec
                + replay_sample_sec
                + policy_search_sec
                + replay_train_sec,
                "readback_sec": readback_sec,
                "total_sec": total_sec,
                "simulations": float(self._simulations),
                "channels": float(self._channels),
                "roots": float(root_count),
                "input_rank": float(len(stack_shape)),
                "input_bytes": float(getattr(observation, "nbytes", 0)),
                "model_eval_count": float(root_count * self._simulations),
                "output_readback_bytes": 8.0 if self._readback_checksum else 0.0,
                "action_mask_consumed": 1.0,
                "compile_excluded_by_warmup": 1.0,
                "resident_stack_h2d_sec": host_to_device_sec,
                "resident_action_mask_h2d_sec": 0.0,
                "resident_metadata_h2d_sec": 0.0,
                "resident_replay_write_sec": replay_write_sec,
                "resident_replay_sample_sec": replay_sample_sec,
                "resident_policy_search_sec": policy_search_sec,
                "resident_replay_train_sec": replay_train_sec,
                "resident_readback_sec": readback_sec,
                "resident_total_sec": total_sec,
                "resident_host_to_device_bytes": host_to_device_bytes,
                "resident_replay_capacity": float(self._capacity),
                "resident_replay_valid_count": float(self._valid_count_host),
                "resident_replay_write_count": float(root_count),
                "resident_sample_batch_size": float(self._sample_batch_size),
                "resident_model_eval_count": float(root_count * self._simulations),
                "resident_replay_train_steps": float(self._replay_train_steps),
                "resident_checksum": checksum_value,
            }
        )

    def _ensure_replay(self, *, root_count: int) -> None:
        if root_count < 1:
            raise ValueError("resident chunk probe requires at least one root")
        capacity = int(root_count * self._replay_steps)
        if (
            self._replay_obs is not None
            and self._replay_mask is not None
            and self._capacity == capacity
            and self._root_count == root_count
        ):
            return
        jnp = self._jnp
        self._capacity = capacity
        self._root_count = root_count
        self._replay_obs = jnp.zeros(
            (capacity, POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE),
            dtype=jnp.uint8,
        )
        self._replay_mask = jnp.zeros((capacity, ACTION_COUNT), dtype=jnp.bool_)
        self._cursor = jnp.asarray(0, dtype=jnp.int32)
        self._valid_count = jnp.asarray(0, dtype=jnp.int32)
        self._cursor_host = 0
        self._valid_count_host = 0
        self._replay_obs.block_until_ready()
        self._replay_mask.block_until_ready()

    def _write_device(
        self,
        replay_obs: Any,
        replay_mask: Any,
        cursor: Any,
        valid_count: Any,
        flat_stack: Any,
        flat_mask: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        jnp = self._jnp
        root_count = int(self._root_count)
        capacity = int(self._capacity)
        replay_obs = self._jax.lax.dynamic_update_slice(
            replay_obs,
            flat_stack,
            (cursor, 0, 0, 0),
        )
        replay_mask = self._jax.lax.dynamic_update_slice(replay_mask, flat_mask, (cursor, 0))
        next_cursor = (cursor + jnp.asarray(root_count, dtype=jnp.int32)) % jnp.asarray(
            capacity,
            dtype=jnp.int32,
        )
        next_valid = jnp.minimum(
            valid_count + jnp.asarray(root_count, dtype=jnp.int32),
            jnp.asarray(capacity, dtype=jnp.int32),
        )
        checksum = jnp.sum(flat_stack[:, -1, :2, :2].astype(jnp.float32))
        return replay_obs, replay_mask, next_cursor, next_valid, checksum

    def _sample_device(
        self,
        replay_obs: Any,
        replay_mask: Any,
        cursor: Any,
        valid_count: Any,
    ) -> tuple[Any, Any, Any]:
        jnp = self._jnp
        valid = jnp.maximum(valid_count, jnp.asarray(1, dtype=jnp.int32))
        offsets = jnp.arange(self._sample_batch_size, dtype=jnp.int32)
        indices = (offsets * jnp.asarray(9973, dtype=jnp.int32) + cursor) % valid
        sample_obs = replay_obs[indices]
        sample_mask = replay_mask[indices]
        checksum = jnp.sum(sample_mask.astype(jnp.float32))
        return sample_obs, sample_mask, checksum

    def _policy_search_device(self, flat_stack: Any, action_mask: Any) -> Any:
        return self._run_conv_loop(
            self._normalize(flat_stack),
            action_mask,
            self._simulations,
        )

    def _replay_train_device(self, sample_obs: Any, sample_mask: Any) -> Any:
        return self._run_conv_loop(
            self._normalize(sample_obs),
            sample_mask,
            self._replay_train_steps,
        )

    def _normalize(self, stack: Any) -> Any:
        jnp = self._jnp
        return stack.astype(jnp.float32) * jnp.asarray(1.0 / 255.0, dtype=jnp.float32)

    def _run_conv_loop(self, flat_stack: Any, action_mask: Any, steps: int) -> Any:
        jnp = self._jnp
        x = jnp.transpose(flat_stack, (0, 2, 3, 1))
        mask = action_mask.reshape((-1, ACTION_COUNT)).astype(jnp.float32)
        acc = jnp.asarray(0.0, dtype=jnp.float32)
        for _ in range(int(steps)):
            y = self._jax.lax.conv_general_dilated(
                x,
                self._kernel,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            y = jnp.tanh(y)
            pooled = jnp.mean(y, axis=(1, 2))
            logits = pooled @ self._policy
            acc = acc + jnp.sum(logits * mask)
            x = jnp.repeat(jnp.mean(y, axis=-1, keepdims=True), POLICY_FRAME_STACK_DEPTH, axis=-1)
        return acc


def _plain_lightzero_value(value: Any) -> Any:
    """Convert LightZero/Torch/JAX-ish outputs to values the row decoder can inspect."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return value.detach().cpu().numpy()
        except Exception:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {key: _plain_lightzero_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_lightzero_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_lightzero_value(item) for item in value]
    return value


def _policy_output_row_from_plain(plain: Any, row: int) -> Any:
    row_int = int(row)
    if isinstance(plain, Mapping):
        for key in (row_int, str(row_int)):
            if key in plain:
                return plain[key]
        row_item: dict[str, Any] = {}
        for key, value in plain.items():
            if isinstance(value, Mapping):
                row_item[key] = _policy_output_row_from_plain(value, row_int)
                continue
            try:
                import numpy as np

                array = np.asarray(value)
            except Exception:
                row_item[key] = value
                continue
            if array.ndim > 0 and array.shape[0] > row_int:
                row_item[key] = array[row_int]
            else:
                row_item[key] = value
        return row_item
    if isinstance(plain, list) and len(plain) > row_int:
        return plain[row_int]
    return plain


def _root_output_from_plain(plain: Any) -> Any:
    if isinstance(plain, Mapping):
        for key in (0, "0"):
            if key in plain:
                return plain[key]
    return plain


def _extract_eval_action_from_plain(plain: Any) -> int:
    import numpy as np

    if isinstance(plain, Mapping):
        if "action" in plain:
            return int(np.asarray(plain["action"]).reshape(-1)[0])
        for key in (0, "0"):
            if key in plain:
                return _extract_eval_action_from_plain(plain[key])
        for key in ("actions", "selected_action", "selected_actions"):
            if key in plain:
                return int(np.asarray(plain[key]).reshape(-1)[0])
    if isinstance(plain, list) and plain:
        return _extract_eval_action_from_plain(plain[0])
    raise ValueError(f"could not extract action from LightZero output row: {plain!r}")


def _root_value_from_policy_output(output: Any) -> float | None:
    import numpy as np

    root = _root_output_from_plain(output)
    if isinstance(root, Mapping):
        for key in ("searched_value", "predicted_value", "value"):
            if key not in root:
                continue
            try:
                return float(np.asarray(root[key]).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                pass
    return None


def _float_field_from_policy_output(output: Any, keys: tuple[str, ...]) -> float | None:
    import numpy as np

    root = _root_output_from_plain(output)
    if not isinstance(root, Mapping):
        return None
    for key in keys:
        if key not in root:
            continue
        try:
            return float(np.asarray(root[key]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            pass
    return None


def _action_vector_field_from_policy_output(
    output: Any,
    keys: tuple[str, ...],
    *,
    np: Any,
    flat_mask_row: Any,
) -> Any | None:
    root = _root_output_from_plain(output)
    if not isinstance(root, Mapping):
        return None
    legal_actions = np.flatnonzero(np.asarray(flat_mask_row) == 1.0)
    for key in keys:
        if key not in root:
            continue
        vector = np.asarray(root[key], dtype=np.float32).reshape(-1)
        if vector.size == ACTION_COUNT:
            return vector
        if vector.size == legal_actions.size:
            full = np.zeros((ACTION_COUNT,), dtype=np.float32)
            full[legal_actions.astype(np.int64)] = vector
            return full
    return None


def _visit_counts_from_policy_output(output: Any) -> Any | None:
    import numpy as np

    root = _root_output_from_plain(output)
    if not isinstance(root, Mapping):
        return None
    for key in ("visit_count_distribution", "visit_count_distributions", "visit_counts"):
        if key not in root:
            continue
        return np.asarray(root[key], dtype=np.float32).reshape(-1)
    return None


def _compact_mcts_arrays_from_lightzero_plain(
    *,
    np: Any,
    plain_output: Any,
    active_root_count: int,
    flat_mask: Any,
) -> dict[str, Any]:
    """Decode stock LightZero policy output into compact arrays for profiling."""

    root_count = int(active_root_count)
    actions = np.zeros((root_count,), dtype=np.int64)
    searched_values = np.zeros((root_count,), dtype=np.float32)
    searched_value_present = np.zeros((root_count,), dtype=np.bool_)
    predicted_values = np.zeros((root_count,), dtype=np.float32)
    predicted_value_present = np.zeros((root_count,), dtype=np.bool_)
    policy_logits = np.zeros((root_count, ACTION_COUNT), dtype=np.float32)
    policy_logits_present = np.zeros((root_count,), dtype=np.bool_)
    visit_distributions = np.zeros((root_count, ACTION_COUNT), dtype=np.float32)
    visit_present_count = 0
    illegal_action_count = 0
    for row in range(root_count):
        row_output = _policy_output_row_from_plain(plain_output, row)
        action = int(_extract_eval_action_from_plain(row_output))
        actions[row] = action
        if action < 0 or action >= ACTION_COUNT or float(flat_mask[row, action]) != 1.0:
            illegal_action_count += 1
        root_value = _root_value_from_policy_output(row_output)
        if root_value is not None:
            searched_values[row] = np.float32(root_value)
            searched_value_present[row] = True
        predicted_value = _float_field_from_policy_output(
            row_output,
            ("predicted_value", "pred_value"),
        )
        if predicted_value is not None:
            predicted_values[row] = np.float32(predicted_value)
            predicted_value_present[row] = True
        logits = _action_vector_field_from_policy_output(
            row_output,
            ("predicted_policy_logits", "policy_logits"),
            np=np,
            flat_mask_row=flat_mask[row],
        )
        if logits is not None:
            policy_logits[row] = logits
            policy_logits_present[row] = True
        visit_array = _visit_counts_from_policy_output(row_output)
        if visit_array is not None:
            if visit_array.size == ACTION_COUNT:
                legal_mask = np.asarray(flat_mask[row]) == 1.0
                visit_distributions[row] = np.where(legal_mask, visit_array, 0.0)
                total = float(visit_distributions[row].sum())
                if total > 0.0:
                    visit_distributions[row] /= np.float32(total)
                visit_present_count += 1
            else:
                legal_actions = np.flatnonzero(np.asarray(flat_mask[row]) == 1.0)
                if visit_array.size == legal_actions.size:
                    visit_distributions[row, legal_actions.astype(np.int64)] = visit_array
                    total = float(visit_distributions[row].sum())
                    if total > 0.0:
                        visit_distributions[row] /= np.float32(total)
                    visit_present_count += 1
    action_checksum = int(
        sum((index + 1) * (int(action) + 1) for index, action in enumerate(actions))
    )
    return {
        "actions": actions,
        "searched_values": searched_values,
        "searched_value_present_mask": searched_value_present,
        "predicted_values": predicted_values,
        "predicted_value_present_mask": predicted_value_present,
        "policy_logits": policy_logits,
        "policy_logits_present_mask": policy_logits_present,
        "visit_distributions": visit_distributions,
        "visit_present_count": int(visit_present_count),
        "illegal_action_count": int(illegal_action_count),
        "action_checksum": int(action_checksum),
    }


def _small_compact_mcts_arrays_debug(
    *,
    np: Any,
    actions: Any,
    searched_values: Any,
    visit_distributions: Any,
    searched_value_present_mask: Any | None = None,
    predicted_values: Any | None = None,
    policy_logits: Any | None = None,
    max_roots: int = 16,
) -> dict[str, Any]:
    """Return compact arrays for tiny parity tests without bloating profile rows."""

    action_array = np.asarray(actions)
    root_count = int(action_array.shape[0]) if action_array.ndim else 0
    if root_count > int(max_roots):
        return {
            "included": False,
            "reason": "root_count_exceeds_debug_cap",
            "root_count": root_count,
            "max_roots": int(max_roots),
        }

    payload: dict[str, Any] = {
        "included": True,
        "root_count": root_count,
        "actions": action_array.astype(int).tolist(),
        "searched_values": np.asarray(searched_values, dtype=np.float32).tolist(),
        "visit_distributions": np.asarray(visit_distributions, dtype=np.float32).tolist(),
    }
    if searched_value_present_mask is not None:
        payload["searched_value_present_mask"] = np.asarray(
            searched_value_present_mask,
            dtype=bool,
        ).tolist()
    if predicted_values is not None:
        payload["predicted_values"] = np.asarray(predicted_values, dtype=np.float32).tolist()
    if policy_logits is not None:
        payload["policy_logits"] = np.asarray(policy_logits, dtype=np.float32).tolist()
    return payload


def _output_key_sample_from_plain(plain: Any, *, limit: int = 24) -> list[str]:
    if isinstance(plain, Mapping):
        return sorted(str(key) for key in plain.keys())[: int(limit)]
    if isinstance(plain, list):
        return [f"list[{len(plain)}]"]
    return [type(plain).__name__]


def _policy_model_device(policy: Any) -> Any:
    import torch

    model = getattr(policy, "_model", None)
    if model is None:
        return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


class _LightZeroModelCallTimer:
    """Profile-only timer for model calls made inside LightZero collect forward."""

    def __init__(self, model: Any, *, device: Any) -> None:
        self._model = model
        self._device = device
        self._originals: dict[str, Any] = {}
        self._timers = {
            "initial_inference": 0.0,
            "recurrent_inference": 0.0,
        }
        self._calls = {
            "initial_inference": 0,
            "recurrent_inference": 0,
        }
        self._status = "not_installed"

    @contextlib.contextmanager
    def patch(self):
        if self._model is None:
            self._status = "model_missing"
            yield self
            return
        installed = 0
        try:
            for name in ("initial_inference", "recurrent_inference"):
                original = getattr(self._model, name, None)
                if not callable(original):
                    continue
                self._originals[name] = original

                def wrapped(*args, _name=name, _original=original, **kwargs):
                    started = time.perf_counter()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        self._sync_if_cuda()
                        self._timers[_name] += time.perf_counter() - started
                        self._calls[_name] += 1

                setattr(self._model, name, wrapped)
                installed += 1
            self._status = "installed" if installed else "no_methods"
            yield self
        except Exception:
            self._status = "patch_failed"
            raise
        finally:
            for name, original in self._originals.items():
                try:
                    setattr(self._model, name, original)
                except Exception:
                    self._status = "restore_failed"
            self._originals.clear()

    def _sync_if_cuda(self) -> None:
        if not str(self._device).startswith("cuda"):
            return
        try:
            import torch

            torch.cuda.synchronize(self._device)
        except Exception:
            self._status = "cuda_sync_failed"

    def summary(self) -> dict[str, Any]:
        model_total_sec = float(sum(self._timers.values()))
        return {
            "status": self._status,
            "initial_inference_sec": float(self._timers["initial_inference"]),
            "initial_inference_calls": int(self._calls["initial_inference"]),
            "recurrent_inference_sec": float(self._timers["recurrent_inference"]),
            "recurrent_inference_calls": int(self._calls["recurrent_inference"]),
            "model_total_sec": model_total_sec,
        }


class _LightZeroCollectForwardInternalTimer:
    """Profile-only timers for the non-model pieces around LightZero collect search."""

    def __init__(self, policy: Any, *, device: Any) -> None:
        self._policy = policy
        self._device = device
        self._restore: list[tuple[Any, str, Any]] = []
        self._timers: dict[str, float] = {
            "mcts_search": 0.0,
            "ctree_batch_traverse": 0.0,
            "ctree_batch_backpropagate": 0.0,
        }
        self._calls: dict[str, int] = {
            "mcts_search": 0,
            "ctree_batch_traverse": 0,
            "ctree_batch_backpropagate": 0,
        }
        self._errors: list[str] = []

    @contextlib.contextmanager
    def patch(self):
        self._patch_policy_mcts_search()
        self._patch_muzero_ctree_functions()
        try:
            yield self
        finally:
            while self._restore:
                obj, name, original = self._restore.pop()
                try:
                    setattr(obj, name, original)
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    self._errors.append(
                        f"restore {type(obj).__name__}.{name} failed: {type(exc).__name__}: {exc}"
                    )

    def _patch_policy_mcts_search(self) -> None:
        mcts = getattr(self._policy, "_mcts_collect", None)
        search = getattr(mcts, "search", None)
        if not callable(search):
            return
        self._patch_callable(mcts, "search", "mcts_search", sync_cuda=True)

    def _patch_muzero_ctree_functions(self) -> None:
        try:
            import importlib

            mcts_ctree = importlib.import_module("lzero.mcts.tree_search.mcts_ctree")
        except Exception as exc:  # pragma: no cover - local tests usually lack lzero.
            self._errors.append(f"import mcts_ctree failed: {type(exc).__name__}: {exc}")
            return
        tree_muzero = getattr(mcts_ctree, "tree_muzero", None)
        if tree_muzero is None:
            self._errors.append("mcts_ctree.tree_muzero missing")
            return
        self._patch_callable(tree_muzero, "batch_traverse", "ctree_batch_traverse")
        self._patch_callable(
            tree_muzero,
            "batch_backpropagate",
            "ctree_batch_backpropagate",
        )

    def _patch_callable(
        self,
        obj: Any,
        name: str,
        metric: str,
        *,
        sync_cuda: bool = False,
    ) -> None:
        original = getattr(obj, name, None)
        if not callable(original):
            return

        def wrapped(*args, _original=original, _metric=metric, **kwargs):
            started = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                if sync_cuda:
                    self._sync_if_cuda()
                self._timers[_metric] += time.perf_counter() - started
                self._calls[_metric] += 1

        try:
            setattr(obj, name, wrapped)
        except Exception as exc:  # pragma: no cover - C extension may be readonly.
            self._errors.append(
                f"patch {type(obj).__name__}.{name} failed: {type(exc).__name__}: {exc}"
            )
            return
        self._restore.append((obj, name, original))

    def _sync_if_cuda(self) -> None:
        if not str(self._device).startswith("cuda"):
            return
        try:
            import torch

            torch.cuda.synchronize(self._device)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            self._errors.append(f"cuda sync failed: {type(exc).__name__}: {exc}")

    def summary(self) -> dict[str, Any]:
        patched_metrics = sum(1 for calls in self._calls.values() if calls > 0)
        return {
            "status": "installed" if patched_metrics else "no_calls",
            "patched_metric_count": int(patched_metrics),
            "errors": list(self._errors),
            "mcts_search_sec": float(self._timers["mcts_search"]),
            "mcts_search_calls": int(self._calls["mcts_search"]),
            "ctree_batch_traverse_sec": float(self._timers["ctree_batch_traverse"]),
            "ctree_batch_traverse_calls": int(self._calls["ctree_batch_traverse"]),
            "ctree_batch_backpropagate_sec": float(self._timers["ctree_batch_backpropagate"]),
            "ctree_batch_backpropagate_calls": int(self._calls["ctree_batch_backpropagate"]),
        }


def _build_profile_lightzero_policy(
    *,
    seed: int,
    use_cuda: bool,
    num_simulations: int,
    collect_with_pure_policy: bool,
    policy_batch_size: int,
    max_ticks: int,
    root_noise_weight: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    import torch
    from ding.config import compile_config
    from lzero.policy.muzero import MuZeroPolicy

    from curvyzero.training import lightzero_config_builder as lz_config

    # Import registers the CurvyTron env type for compile_config.
    import curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env  # noqa: F401

    cuda = bool(use_cuda and torch.cuda.is_available())
    patched = lz_config.build_visual_survival_configs(
        seed=int(seed),
        exp_name="profile_only_lightzero_collect_forward_consumer",
        telemetry_path="/tmp/curvyzero_profile_lightzero_collect_forward.jsonl",
        cuda=cuda,
        max_env_step=max(1, int(max_ticks)),
        source_max_steps=max(1, int(max_ticks)),
        decision_ms=float(lz_config.DEFAULT_DECISION_MS),
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=int(num_simulations),
        batch_size=max(2, int(policy_batch_size)),
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=1,
        save_ckpt_after_iter=1_000_000,
        env_variant=lz_config.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=lz_config.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
        reward_outcome_alpha=float(lz_config.DEFAULT_REWARD_OUTCOME_ALPHA),
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id="none",
        disable_death_for_profile=True,
        env_telemetry_stride=1,
        env_manager_type="base",
        opponent_policy_kind=lz_config.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        decision_source_frames=int(lz_config.DEFAULT_DECISION_SOURCE_FRAMES),
        source_physics_step_ms=float(lz_config.DEFAULT_SOURCE_PHYSICS_STEP_MS),
        source_max_steps_semantics="source_physics_steps",
        source_state_trail_render_mode=lz_config.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
        source_state_bonus_render_mode=lz_config.DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
        policy_observation_backend=lz_config.DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
        learner_seat_mode=lz_config.DEFAULT_LEARNER_SEAT_MODE,
        policy_action_repeat_min=int(lz_config.DEFAULT_POLICY_ACTION_REPEAT_MIN),
        policy_action_repeat_max=int(lz_config.DEFAULT_POLICY_ACTION_REPEAT_MAX),
        policy_action_repeat_extra_probability=float(
            lz_config.DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
        ),
        natural_bonus_spawn=bool(lz_config.DEFAULT_NATURAL_BONUS_SPAWN),
        profile_env_timing_enabled=True,
        opponent_death_mode=lz_config.DEFAULT_OPPONENT_DEATH_MODE,
        opponent_runtime_mode=lz_config.DEFAULT_OPPONENT_RUNTIME_MODE,
        model_support_cap=lz_config.DEFAULT_MODEL_SUPPORT_CAP,
        td_steps=lz_config.DEFAULT_TD_STEPS,
        exploration_bonus=None,
    )
    cfg = compile_config(
        copy.deepcopy(patched["main_config"]),
        seed=int(seed),
        auto=True,
        create_cfg=copy.deepcopy(patched["create_config"]),
        save_cfg=False,
    )
    cfg.policy.cuda = cuda
    cfg.policy.device = "cuda" if cuda else "cpu"
    cfg.policy.collect_with_pure_policy = bool(collect_with_pure_policy)
    if root_noise_weight is not None:
        cfg.policy.root_noise_weight = float(root_noise_weight)
    policy = MuZeroPolicy(cfg.policy)
    model = getattr(policy, "_model", None)
    if model is not None and hasattr(model, "eval"):
        model.eval()
    return {
        "policy": policy,
        "build_sec": time.perf_counter() - started,
        "requested_cuda": bool(use_cuda),
        "cuda": cuda,
        "policy_class": f"{type(policy).__module__}.{type(policy).__name__}",
        "model_class": f"{type(model).__module__}.{type(model).__name__}"
        if model is not None
        else None,
        "policy_device": str(_policy_model_device(policy)),
        "num_simulations": int(getattr(cfg.policy, "num_simulations", num_simulations)),
        "collect_with_pure_policy": bool(
            getattr(cfg.policy, "collect_with_pure_policy", collect_with_pure_policy)
        ),
        "root_noise_weight": float(getattr(cfg.policy, "root_noise_weight", 0.25)),
        "batch_size": int(getattr(cfg.policy, "batch_size", policy_batch_size)),
        "surface": {
            "env_variant": patched["surface"].get("env_variant"),
            "observation_shape": patched["surface"].get("observation_shape"),
            "policy_observation_backend": patched["surface"].get("policy_observation_backend"),
            "policy_trail_render_mode": patched["surface"].get("policy_trail_render_mode"),
            "policy_bonus_render_mode": patched["surface"].get("policy_bonus_render_mode"),
        },
    }


def _compact_root_batch_grid_inputs(root_batch: Any) -> tuple[Any, Any]:
    import numpy as np

    batch_size = int(root_batch.metadata.get("batch_size", 0))
    player_count = int(root_batch.metadata.get("player_count", 0))
    if batch_size <= 0 or player_count <= 0:
        raise ValueError("compact root batch metadata must include batch_size/player_count")
    observation = np.asarray(root_batch.observation)
    legal_mask = np.asarray(root_batch.legal_mask)
    expected_root_count = int(batch_size * player_count)
    if observation.shape[0] != expected_root_count:
        raise ValueError("compact root observation count does not match metadata")
    if legal_mask.shape != (expected_root_count, ACTION_COUNT):
        raise ValueError("compact root legal_mask must have shape [root,3]")
    return (
        observation.reshape(batch_size, player_count, *observation.shape[1:]),
        legal_mask.reshape(batch_size, player_count, ACTION_COUNT),
    )


def _compact_search_profile_telemetry(telemetry: Any) -> dict[str, Any]:
    """Keep small profile telemetry fields for slab summaries.

    The direct CTree probes also carry debug arrays. Those are useful in local
    tests, but too bulky to nest inside every compact search result.
    """

    def plain(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                pass
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return str(value)

    if not isinstance(telemetry, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in telemetry.items():
        name = str(key)
        if "debug_arrays" in name:
            continue
        out[name] = plain(value)
    return out


class _LightZeroCollectForwardCompactSearchService(CompactSearchServiceV1):
    """Adapter from compact root batches to the existing direct CTree probe."""

    def __init__(self, probe: Any) -> None:
        self._probe = probe
        self.search_impl = str(probe._arrays_boundary_impl)
        self.num_simulations = int(probe._num_simulations)

    def run(self, root_batch: Any) -> Any:
        obs_grid, mask_grid = _compact_root_batch_grid_inputs(root_batch)
        self._probe._last_direct_mcts_arrays = None
        probe_result = self._probe.run(obs_grid, mask_grid)
        search_arrays = self._probe._last_direct_mcts_arrays
        if search_arrays is None:
            raise ValueError("compact search service requires direct CTree compact arrays")
        profile_telemetry = _compact_search_profile_telemetry(
            getattr(probe_result, "telemetry", {})
        )
        return compact_search_result_v1_from_arrays(
            root_batch,
            search_arrays,
            default_search_impl=self.search_impl,
            default_num_simulations=self.num_simulations,
            metadata={
                "profile_backend": str(getattr(self._probe, "backend_name", "")),
                "profile_semantics": str(getattr(self._probe, "semantics", "")),
                "compact_search_service_adapter": True,
                "profile_telemetry": profile_telemetry,
            },
        )


class _LightZeroArrayCeilingCompactSearchService(CompactSearchServiceV1):
    """Adapter from compact root batches to array-ceiling compact probes."""

    def __init__(self, probe: Any) -> None:
        self._probe = probe
        self.search_impl = str(probe._mode)
        self.num_simulations = int(probe._num_simulations)
        self.supports_two_phase_compact_search = self.search_impl in {
            LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE,
            LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER,
        }

    def run(self, root_batch: Any) -> Any:
        if self.search_impl == LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE:
            return self._probe._run_compact_torch_search_service_root_batch(root_batch)
        if self.search_impl == LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER:
            return self._probe._run_fixed_shape_search_owner_root_batch(root_batch)
        obs_grid, mask_grid = _compact_root_batch_grid_inputs(root_batch)
        self._probe._last_compact_search_arrays = None
        probe_result = self._probe.run(obs_grid, mask_grid)
        search_arrays = self._probe._last_compact_search_arrays
        if search_arrays is None:
            raise ValueError("compact search service requires array-ceiling compact arrays")
        profile_telemetry = _compact_search_profile_telemetry(
            getattr(probe_result, "telemetry", {})
        )
        return compact_search_result_v1_from_arrays(
            root_batch,
            search_arrays,
            default_search_impl=self.search_impl,
            default_num_simulations=self.num_simulations,
            metadata={
                "profile_backend": str(getattr(self._probe, "backend_name", "")),
                "profile_semantics": str(getattr(self._probe, "semantics", "")),
                "compact_search_service_adapter": True,
                "array_ceiling_mode": self.search_impl,
                "profile_telemetry": profile_telemetry,
            },
        )

    def run_action_step(self, root_batch: Any) -> Any:
        if self.search_impl == LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE:
            return self._probe._run_compact_torch_search_service_action_step(root_batch)
        if self.search_impl != LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER:
            raise ValueError("two-phase compact search is only exposed for compact owner modes")
        return self._probe._run_fixed_shape_search_owner_action_step(root_batch)

    def flush_replay_payload(self, replay_payload_handle: str) -> Any:
        if self.search_impl == LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE:
            return self._probe._flush_compact_torch_search_service_replay_payload(
                replay_payload_handle
            )
        if self.search_impl != LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER:
            raise ValueError("two-phase compact search is only exposed for compact owner modes")
        return self._probe._flush_fixed_shape_search_owner_replay_payload(replay_payload_handle)

    def flush_device_replay_payload(self, replay_payload_handle: str) -> Any:
        if self.search_impl != LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE:
            raise ValueError(
                "device replay payloads are only exposed for compact_torch_search_service"
            )
        return self._probe._flush_compact_torch_search_service_device_replay_payload(
            replay_payload_handle
        )


class _LightZeroCollectForwardStackProbe:
    """Profile-only real LightZero collect-forward consumer for pre-scalar stacks."""

    backend_name = "lightzero_collect_forward_consumer"
    semantics = "lightzero_collect_forward_search_cpu_tree"
    fixed_opponent_to_play = -1

    def __init__(
        self,
        *,
        policy: Any,
        policy_metadata: Mapping[str, Any],
        num_simulations: int,
        temperature: float,
        epsilon: float,
        arrays_boundary: bool = False,
        arrays_boundary_impl: str = LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_STOCK_FACADE,
        input_mode: str = LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8,
    ) -> None:
        self._policy = policy
        self._policy_metadata = dict(policy_metadata)
        self._arrays_boundary = bool(arrays_boundary)
        self._arrays_boundary_impl = str(arrays_boundary_impl)
        self._input_mode = str(input_mode)
        if self._arrays_boundary_impl not in LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPLS:
            allowed = ", ".join(LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPLS)
            raise ValueError(
                f"arrays_boundary_impl must be one of {allowed}; got {self._arrays_boundary_impl!r}"
            )
        if self._input_mode not in LIGHTZERO_ARRAY_CEILING_INPUT_MODES:
            allowed = ", ".join(LIGHTZERO_ARRAY_CEILING_INPUT_MODES)
            raise ValueError(f"input_mode must be one of {allowed}; got {self._input_mode!r}")
        if self._arrays_boundary:
            if self._arrays_boundary_impl in {
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT,
            }:
                if (
                    self._arrays_boundary_impl
                    == LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT
                ):
                    self.backend_name = (
                        "lightzero_mcts_arrays_direct_ctree_gpu_latent_precomputed_consumer"
                    )
                    self.semantics = (
                        "lightzero_mcts_arrays_direct_ctree_gpu_latent_precomputed_"
                        "recurrent_profile_not_model"
                    )
                else:
                    self.backend_name = "lightzero_mcts_arrays_direct_ctree_gpu_latent_consumer"
                    self.semantics = LIGHTZERO_MCTS_ARRAYS_BOUNDARY_GPU_LATENT_SEMANTICS
            elif self._arrays_boundary_impl == LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE:
                self.backend_name = "lightzero_mcts_arrays_direct_ctree_consumer"
                self.semantics = LIGHTZERO_MCTS_ARRAYS_BOUNDARY_DIRECT_SEMANTICS
            else:
                self.backend_name = "lightzero_mcts_arrays_boundary_consumer"
                self.semantics = LIGHTZERO_MCTS_ARRAYS_BOUNDARY_SEMANTICS
        self._num_simulations = _positive_int(
            num_simulations,
            "hybrid_lightzero_consumer_num_simulations",
        )
        self._temperature = float(temperature)
        self._epsilon = float(epsilon)
        self._collect_with_pure_policy = bool(
            self._policy_metadata.get("collect_with_pure_policy", False)
        )
        self._resident_obs_tensor: Any | None = None
        self._resident_obs_signature: tuple[Any, ...] | None = None
        self._last_direct_mcts_arrays: dict[str, Any] | None = None
        self._last_compact_service_root_batch: Any | None = None
        self._last_compact_service_search_result: Any | None = None
        self._last_compact_service_replay_chunk: Any | None = None
        if self._temperature <= 0.0:
            raise ValueError("hybrid_lightzero_consumer_temperature must be positive")
        if not 0.0 <= self._epsilon <= 1.0:
            raise ValueError("hybrid_lightzero_consumer_epsilon must be in [0, 1]")

    def run(self, observation: Any, action_mask: Any) -> HybridBatchedStackProbeResult:
        import numpy as np
        import torch

        self._last_direct_mcts_arrays = None
        stack = np.asarray(observation)
        stack_shape = tuple(int(dim) for dim in stack.shape)
        if stack_shape[2:] != (POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE):
            raise ValueError(
                "LightZero collect-forward probe expects observation shape "
                f"[B,P,{POLICY_FRAME_STACK_DEPTH},{TARGET_SIZE},{TARGET_SIZE}], got {stack_shape}"
            )
        mask = np.asarray(action_mask, dtype=np.float32)
        expected_mask_shape = stack_shape[:2] + (ACTION_COUNT,)
        if tuple(int(dim) for dim in mask.shape) != expected_mask_shape:
            raise ValueError(
                "LightZero collect-forward probe action_mask shape mismatch; "
                f"got {mask.shape}, expected {expected_mask_shape}"
            )
        if not bool(np.all((mask == 0.0) | (mask == 1.0))):
            raise ValueError(
                "LightZero collect-forward probe action_mask must be binary 0/1 "
                "to match stock LightZero legal-action semantics"
            )
        total_root_count = int(stack_shape[0] * stack_shape[1])
        rows_all = np.repeat(np.arange(stack_shape[0], dtype=np.int64), stack_shape[1])
        players_all = np.tile(np.arange(stack_shape[1], dtype=np.int64), stack_shape[0])
        flat_mask_all = mask.reshape((total_root_count, ACTION_COUNT))
        legal_root_mask = np.any(flat_mask_all == 1.0, axis=1)
        active_root_count = int(np.count_nonzero(legal_root_mask))
        dropped_zero_mask_root_count = int(total_root_count - active_root_count)
        rows = rows_all[legal_root_mask]
        players = players_all[legal_root_mask]
        flat_mask = flat_mask_all[legal_root_mask]
        ready_env_id = np.arange(active_root_count, dtype=np.int64)
        to_play = [self.fixed_opponent_to_play] * active_root_count
        device = _policy_model_device(self._policy)

        started = time.perf_counter()
        flat_stack_all = stack.reshape(
            (total_root_count, POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE)
        )
        flat_stack = flat_stack_all[legal_root_mask]
        tensor_prepare_sec = time.perf_counter() - started

        if active_root_count == 0:
            arrays_semantics = self.semantics if self._arrays_boundary else "disabled"
            arrays_impl = self._arrays_boundary_impl if self._arrays_boundary else "disabled"
            arrays_input_mode = self._input_mode if self._arrays_boundary else "disabled"
            return HybridBatchedStackProbeResult(
                telemetry={
                    "host_to_device_sec": 0.0,
                    "host_to_device_bytes": 0.0,
                    "input_transfer_bytes": 0.0,
                    "normalize_sec": 0.0,
                    "device_sec": 0.0,
                    "readback_sec": 0.0,
                    "total_sec": tensor_prepare_sec,
                    "simulations": float(self._num_simulations),
                    "channels": 0.0,
                    "roots": 0.0,
                    "input_rank": float(len(stack_shape)),
                    "input_bytes": float(getattr(stack, "nbytes", 0)),
                    "model_eval_count": 0.0,
                    "output_readback_bytes": 0.0,
                    "action_mask_consumed": 1.0,
                    "compile_excluded_by_warmup": 1.0,
                    "lightzero_consumer_total_sec": tensor_prepare_sec,
                    "lightzero_consumer_tensor_prepare_sec": tensor_prepare_sec,
                    "lightzero_consumer_h2d_sec": 0.0,
                    "lightzero_consumer_normalize_sec": 0.0,
                    "lightzero_consumer_collect_forward_sec": 0.0,
                    "lightzero_consumer_collect_forward_wall_sec": 0.0,
                    "lightzero_consumer_decode_sec": 0.0,
                    "lightzero_consumer_readback_sec": 0.0,
                    "lightzero_consumer_stack_h2d_bytes": 0.0,
                    "lightzero_consumer_input_transfer_bytes": 0.0,
                    "lightzero_consumer_mask_numpy_bytes": float(
                        getattr(flat_mask_all, "nbytes", 0)
                    ),
                    "lightzero_consumer_input_bytes": float(getattr(stack, "nbytes", 0))
                    + float(getattr(mask, "nbytes", 0)),
                    "lightzero_consumer_output_bytes": 0.0,
                    "lightzero_consumer_policy_device": str(device),
                    "lightzero_consumer_policy_class": str(
                        self._policy_metadata.get("policy_class", type(self._policy).__name__)
                    ),
                    "lightzero_consumer_policy_surface": dict(
                        self._policy_metadata.get("surface", {})
                    ),
                    "lightzero_consumer_num_simulations": float(self._num_simulations),
                    "lightzero_consumer_collect_with_pure_policy": bool(
                        self._policy_metadata.get("collect_with_pure_policy", False)
                    ),
                    "lightzero_consumer_input_mode": self._input_mode,
                    "lightzero_consumer_input_freshness": "no_active_roots",
                    "lightzero_consumer_pin_memory_sec": 0.0,
                    "lightzero_consumer_host_prenormalize_sec": 0.0,
                    "lightzero_consumer_input_prepare_sec": 0.0,
                    "lightzero_consumer_resident_first_fill_sec": 0.0,
                    "lightzero_consumer_resident_reused": 0.0,
                    "lightzero_consumer_cpu_tree_included": 0.0
                    if self._collect_with_pure_policy
                    else 1.0,
                    "lightzero_policy_forward_calls": 0.0,
                    "lightzero_total_root_count": float(total_root_count),
                    "lightzero_filtered_zero_mask_root_count": float(dropped_zero_mask_root_count),
                    "lightzero_root_count": 0.0,
                    "lightzero_roots_per_call": 0.0,
                    "lightzero_ready_env_id_first": -1,
                    "lightzero_ready_env_id_last": -1,
                    "lightzero_to_play_mode": "fixed_opponent_minus_one",
                    "lightzero_to_play_sample": [],
                    "lightzero_output_key_sample": [],
                    "lightzero_action_checksum": 0.0,
                    "lightzero_illegal_action_count": 0.0,
                    "lightzero_visit_distribution_count": 0.0,
                    "lightzero_root_value_count": 0.0,
                    "lightzero_root_value_mean": 0.0,
                    "lightzero_first_actions": [],
                    "lightzero_rows_sample": [],
                    "lightzero_players_sample": [],
                    "lightzero_policy_build_sec": float(
                        self._policy_metadata.get("build_sec", 0.0)
                    ),
                    "lightzero_policy_requested_cuda": bool(
                        self._policy_metadata.get("requested_cuda", False)
                    ),
                    "lightzero_policy_cuda": bool(self._policy_metadata.get("cuda", False)),
                    "lightzero_mcts_arrays_boundary_enabled": bool(self._arrays_boundary),
                    "lightzero_mcts_arrays_boundary_semantics": arrays_semantics,
                    "lightzero_mcts_arrays_boundary_impl": arrays_impl,
                    "lightzero_mcts_arrays_boundary_input_mode": arrays_input_mode,
                    "lightzero_mcts_arrays_boundary_input_freshness": "no_active_roots",
                    "lightzero_mcts_arrays_boundary_pin_memory_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_host_prenormalize_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_input_prepare_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_input_transfer_bytes": 0.0,
                    "lightzero_mcts_arrays_boundary_resident_first_fill_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_resident_reused": 0.0,
                    "lightzero_mcts_arrays_boundary_total_sec": (
                        tensor_prepare_sec if self._arrays_boundary else 0.0
                    ),
                    "lightzero_mcts_arrays_boundary_collect_forward_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_decode_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_initial_inference_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_model_output_d2h_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_model_output_d2h_bytes": 0.0,
                    "lightzero_mcts_arrays_boundary_root_prepare_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_search_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_output_assembly_sec": 0.0,
                    "lightzero_mcts_arrays_boundary_compact_output_bytes": 0.0,
                    "lightzero_mcts_arrays_boundary_public_output_bytes": 0.0,
                    "lightzero_mcts_arrays_boundary_action_shape": [0],
                    "lightzero_mcts_arrays_boundary_debug_arrays": {
                        "included": True,
                        "root_count": 0,
                        "actions": [],
                        "searched_values": [],
                        "visit_distributions": [],
                    },
                    "lightzero_mcts_arrays_boundary_visit_shape": [0, ACTION_COUNT],
                    "lightzero_mcts_arrays_boundary_searched_value_shape": [0],
                    "lightzero_mcts_arrays_boundary_visit_present_count": 0.0,
                    "lightzero_mcts_arrays_boundary_root_value_count": 0.0,
                }
            )

        started = time.perf_counter()
        (
            obs_tensor,
            h2d_sec,
            normalize_sec,
            input_telemetry,
        ) = self._prepare_observation_tensor(
            np=np,
            torch=torch,
            flat_stack=flat_stack,
            stack_dtype=stack.dtype,
            device=device,
        )

        if self._arrays_boundary and self._arrays_boundary_impl in {
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT,
        }:
            return self._run_direct_mcts_arrays(
                np=np,
                torch=torch,
                stack=stack,
                mask=mask,
                flat_stack=flat_stack,
                flat_mask=flat_mask,
                obs_tensor=obs_tensor,
                input_telemetry=input_telemetry,
                tensor_prepare_sec=tensor_prepare_sec,
                h2d_sec=h2d_sec,
                normalize_sec=normalize_sec,
                total_root_count=total_root_count,
                active_root_count=active_root_count,
                dropped_zero_mask_root_count=dropped_zero_mask_root_count,
                rows=rows,
                players=players,
                to_play=to_play,
                device=device,
                keep_latents_on_device=(
                    self._arrays_boundary_impl
                    in {
                        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
                        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT,
                    }
                ),
                precompute_recurrent_outputs=(
                    self._arrays_boundary_impl
                    == LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT
                ),
            )

        model_timer = _LightZeroModelCallTimer(
            getattr(self._policy, "_model", None),
            device=device,
        )
        internal_timer = _LightZeroCollectForwardInternalTimer(self._policy, device=device)
        started = time.perf_counter()
        with torch.no_grad(), model_timer.patch(), internal_timer.patch():
            output = self._policy.collect_mode.forward(
                obs_tensor,
                action_mask=flat_mask,
                temperature=self._temperature,
                to_play=to_play,
                epsilon=self._epsilon,
                ready_env_id=ready_env_id,
            )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        collect_forward_sec = time.perf_counter() - started
        model_timer_summary = model_timer.summary()
        internal_timer_summary = internal_timer.summary()
        collect_forward_model_sec = float(model_timer_summary["model_total_sec"])
        collect_forward_residual_sec = max(0.0, collect_forward_sec - collect_forward_model_sec)
        mcts_search_sec = float(internal_timer_summary["mcts_search_sec"])
        recurrent_model_sec = float(model_timer_summary["recurrent_inference_sec"])
        initial_model_sec = float(model_timer_summary["initial_inference_sec"])
        mcts_non_model_sec = max(0.0, mcts_search_sec - recurrent_model_sec)
        outside_mcts_sec = max(0.0, collect_forward_sec - mcts_search_sec)
        outside_mcts_non_initial_model_sec = max(0.0, outside_mcts_sec - initial_model_sec)

        started = time.perf_counter()
        plain_output = _plain_lightzero_value(output)
        compact_arrays = _compact_mcts_arrays_from_lightzero_plain(
            np=np,
            plain_output=plain_output,
            active_root_count=active_root_count,
            flat_mask=flat_mask,
        )
        actions_np = compact_arrays["actions"]
        searched_values_np = compact_arrays["searched_values"]
        predicted_values_np = compact_arrays["predicted_values"]
        policy_logits_np = compact_arrays["policy_logits"]
        visits_np = compact_arrays["visit_distributions"]
        actions = actions_np.astype(int).tolist()
        root_values = (
            searched_values_np[compact_arrays["searched_value_present_mask"]].astype(float).tolist()
        )
        visit_present_count = int(compact_arrays["visit_present_count"])
        illegal_action_count = int(compact_arrays["illegal_action_count"])
        decode_sec = time.perf_counter() - started

        started = time.perf_counter()
        try:
            output_bytes = len(pickle.dumps(plain_output, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            output_bytes = 0
        readback_sec = time.perf_counter() - started
        action_checksum = int(compact_arrays["action_checksum"])
        compact_output_bytes = int(
            actions_np.nbytes
            + searched_values_np.nbytes
            + predicted_values_np.nbytes
            + policy_logits_np.nbytes
            + visits_np.nbytes
        )
        debug_arrays = _small_compact_mcts_arrays_debug(
            np=np,
            actions=actions_np,
            searched_values=searched_values_np,
            searched_value_present_mask=compact_arrays["searched_value_present_mask"],
            predicted_values=predicted_values_np,
            policy_logits=policy_logits_np,
            visit_distributions=visits_np,
        )
        input_preprocess_sec = float(input_telemetry.get("input_prepare_sec", 0.0))
        input_transfer_bytes = float(
            input_telemetry.get("transfer_bytes", getattr(flat_stack, "nbytes", 0))
        )
        input_freshness = str(input_telemetry.get("input_freshness", "fresh"))
        if illegal_action_count:
            raise ValueError(
                "LightZero collect-forward probe decoded illegal actions: "
                f"{illegal_action_count} / {active_root_count}"
            )
        total_sec = (
            tensor_prepare_sec
            + input_preprocess_sec
            + h2d_sec
            + normalize_sec
            + collect_forward_sec
            + decode_sec
            + readback_sec
        )
        return HybridBatchedStackProbeResult(
            telemetry={
                "host_to_device_sec": h2d_sec,
                "host_to_device_bytes": input_transfer_bytes,
                "input_transfer_bytes": input_transfer_bytes,
                "normalize_sec": normalize_sec,
                "device_sec": collect_forward_sec,
                "readback_sec": readback_sec,
                "total_sec": total_sec,
                "simulations": float(self._num_simulations),
                "channels": 0.0,
                "roots": float(active_root_count),
                "input_rank": float(len(stack_shape)),
                "input_bytes": float(getattr(stack, "nbytes", 0)),
                "model_eval_count": float(
                    active_root_count
                    if self._collect_with_pure_policy
                    else active_root_count * (1 + self._num_simulations)
                ),
                "output_readback_bytes": float(output_bytes),
                "action_mask_consumed": 1.0,
                "compile_excluded_by_warmup": 1.0,
                "lightzero_consumer_total_sec": total_sec,
                "lightzero_consumer_tensor_prepare_sec": tensor_prepare_sec,
                "lightzero_consumer_h2d_sec": h2d_sec,
                "lightzero_consumer_normalize_sec": normalize_sec,
                "lightzero_consumer_input_mode": self._input_mode,
                "lightzero_consumer_input_freshness": input_freshness,
                "lightzero_consumer_pin_memory_sec": float(
                    input_telemetry.get("pin_memory_sec", 0.0)
                ),
                "lightzero_consumer_host_prenormalize_sec": float(
                    input_telemetry.get("host_prenormalize_sec", 0.0)
                ),
                "lightzero_consumer_input_prepare_sec": input_preprocess_sec,
                "lightzero_consumer_resident_first_fill_sec": float(
                    input_telemetry.get("resident_first_fill_sec", 0.0)
                ),
                "lightzero_consumer_resident_reused": float(
                    input_telemetry.get("resident_reused", 0.0)
                ),
                "lightzero_consumer_collect_forward_sec": collect_forward_sec,
                "lightzero_consumer_collect_forward_wall_sec": collect_forward_sec,
                "lightzero_consumer_model_initial_inference_sec": float(
                    model_timer_summary["initial_inference_sec"]
                ),
                "lightzero_consumer_model_initial_inference_calls": float(
                    model_timer_summary["initial_inference_calls"]
                ),
                "lightzero_consumer_model_recurrent_inference_sec": float(
                    model_timer_summary["recurrent_inference_sec"]
                ),
                "lightzero_consumer_model_recurrent_inference_calls": float(
                    model_timer_summary["recurrent_inference_calls"]
                ),
                "lightzero_consumer_model_total_sec": collect_forward_model_sec,
                "lightzero_consumer_collect_forward_non_model_sec": (collect_forward_residual_sec),
                "lightzero_consumer_model_timer_status": str(model_timer_summary["status"]),
                "lightzero_consumer_mcts_timer_status": str(internal_timer_summary["status"]),
                "lightzero_consumer_mcts_timer_patched_metric_count": float(
                    internal_timer_summary["patched_metric_count"]
                ),
                "lightzero_consumer_mcts_timer_error_count": float(
                    len(internal_timer_summary["errors"])
                ),
                "lightzero_consumer_mcts_timer_errors": internal_timer_summary["errors"],
                "lightzero_consumer_mcts_search_sec": mcts_search_sec,
                "lightzero_consumer_mcts_search_calls": float(
                    internal_timer_summary["mcts_search_calls"]
                ),
                "lightzero_consumer_mcts_search_non_model_sec": mcts_non_model_sec,
                "lightzero_consumer_collect_forward_outside_mcts_sec": outside_mcts_sec,
                "lightzero_consumer_collect_forward_outside_mcts_non_initial_model_sec": (
                    outside_mcts_non_initial_model_sec
                ),
                "lightzero_consumer_ctree_batch_traverse_sec": float(
                    internal_timer_summary["ctree_batch_traverse_sec"]
                ),
                "lightzero_consumer_ctree_batch_traverse_calls": float(
                    internal_timer_summary["ctree_batch_traverse_calls"]
                ),
                "lightzero_consumer_ctree_batch_backpropagate_sec": float(
                    internal_timer_summary["ctree_batch_backpropagate_sec"]
                ),
                "lightzero_consumer_ctree_batch_backpropagate_calls": float(
                    internal_timer_summary["ctree_batch_backpropagate_calls"]
                ),
                "lightzero_consumer_decode_sec": decode_sec,
                "lightzero_consumer_readback_sec": readback_sec,
                "lightzero_consumer_stack_h2d_bytes": input_transfer_bytes,
                "lightzero_consumer_input_transfer_bytes": input_transfer_bytes,
                "lightzero_consumer_mask_numpy_bytes": float(getattr(flat_mask, "nbytes", 0)),
                "lightzero_consumer_input_bytes": float(getattr(stack, "nbytes", 0))
                + float(getattr(mask, "nbytes", 0)),
                "lightzero_consumer_output_bytes": float(output_bytes),
                "lightzero_consumer_policy_device": str(device),
                "lightzero_consumer_policy_class": str(
                    self._policy_metadata.get("policy_class", type(self._policy).__name__)
                ),
                "lightzero_consumer_policy_surface": dict(self._policy_metadata.get("surface", {})),
                "lightzero_consumer_num_simulations": float(self._num_simulations),
                "lightzero_consumer_collect_with_pure_policy": bool(
                    self._policy_metadata.get("collect_with_pure_policy", False)
                ),
                "lightzero_consumer_cpu_tree_included": 0.0
                if self._collect_with_pure_policy
                else 1.0,
                "lightzero_policy_forward_calls": 1.0,
                "lightzero_total_root_count": float(total_root_count),
                "lightzero_filtered_zero_mask_root_count": float(dropped_zero_mask_root_count),
                "lightzero_root_count": float(active_root_count),
                "lightzero_roots_per_call": float(active_root_count),
                "lightzero_ready_env_id_first": (int(ready_env_id[0]) if active_root_count else -1),
                "lightzero_ready_env_id_last": (int(ready_env_id[-1]) if active_root_count else -1),
                "lightzero_to_play_mode": "fixed_opponent_minus_one",
                "lightzero_to_play_sample": to_play[: min(8, len(to_play))],
                "lightzero_output_key_sample": _output_key_sample_from_plain(plain_output),
                "lightzero_action_checksum": float(action_checksum),
                "lightzero_illegal_action_count": float(illegal_action_count),
                "lightzero_visit_distribution_count": float(visit_present_count),
                "lightzero_root_value_count": float(len(root_values)),
                "lightzero_root_value_mean": (
                    float(sum(root_values) / len(root_values)) if root_values else 0.0
                ),
                "lightzero_first_actions": actions[: min(16, len(actions))],
                "lightzero_rows_sample": rows[: min(8, len(rows))].astype(int).tolist(),
                "lightzero_players_sample": players[: min(8, len(players))].astype(int).tolist(),
                "lightzero_policy_build_sec": float(self._policy_metadata.get("build_sec", 0.0)),
                "lightzero_policy_requested_cuda": bool(
                    self._policy_metadata.get("requested_cuda", False)
                ),
                "lightzero_policy_cuda": bool(self._policy_metadata.get("cuda", False)),
                "lightzero_mcts_arrays_boundary_enabled": bool(self._arrays_boundary),
                "lightzero_mcts_arrays_boundary_semantics": (
                    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_SEMANTICS
                    if self._arrays_boundary
                    else "disabled"
                ),
                "lightzero_mcts_arrays_boundary_impl": (
                    self._arrays_boundary_impl if self._arrays_boundary else "disabled"
                ),
                "lightzero_mcts_arrays_boundary_input_mode": (
                    self._input_mode if self._arrays_boundary else "disabled"
                ),
                "lightzero_mcts_arrays_boundary_pin_memory_sec": (
                    float(input_telemetry.get("pin_memory_sec", 0.0))
                    if self._arrays_boundary
                    else 0.0
                ),
                "lightzero_mcts_arrays_boundary_host_prenormalize_sec": (
                    float(input_telemetry.get("host_prenormalize_sec", 0.0))
                    if self._arrays_boundary
                    else 0.0
                ),
                "lightzero_mcts_arrays_boundary_input_prepare_sec": (
                    input_preprocess_sec if self._arrays_boundary else 0.0
                ),
                "lightzero_mcts_arrays_boundary_input_transfer_bytes": (
                    input_transfer_bytes if self._arrays_boundary else 0.0
                ),
                "lightzero_mcts_arrays_boundary_resident_first_fill_sec": (
                    float(input_telemetry.get("resident_first_fill_sec", 0.0))
                    if self._arrays_boundary
                    else 0.0
                ),
                "lightzero_mcts_arrays_boundary_resident_reused": (
                    float(input_telemetry.get("resident_reused", 0.0))
                    if self._arrays_boundary
                    else 0.0
                ),
                "lightzero_mcts_arrays_boundary_total_sec": (
                    total_sec if self._arrays_boundary else 0.0
                ),
                "lightzero_mcts_arrays_boundary_collect_forward_sec": (
                    collect_forward_sec if self._arrays_boundary else 0.0
                ),
                "lightzero_mcts_arrays_boundary_decode_sec": (
                    decode_sec if self._arrays_boundary else 0.0
                ),
                "lightzero_mcts_arrays_boundary_initial_inference_sec": 0.0,
                "lightzero_mcts_arrays_boundary_root_prepare_sec": 0.0,
                "lightzero_mcts_arrays_boundary_search_sec": 0.0,
                "lightzero_mcts_arrays_boundary_output_assembly_sec": 0.0,
                "lightzero_mcts_arrays_boundary_compact_output_bytes": (
                    float(compact_output_bytes) if self._arrays_boundary else 0.0
                ),
                "lightzero_mcts_arrays_boundary_public_output_bytes": (
                    float(output_bytes) if self._arrays_boundary else 0.0
                ),
                "lightzero_mcts_arrays_boundary_action_shape": (
                    list(actions_np.shape) if self._arrays_boundary else []
                ),
                "lightzero_mcts_arrays_boundary_debug_arrays": (
                    debug_arrays if self._arrays_boundary else {}
                ),
                "lightzero_mcts_arrays_boundary_visit_shape": (
                    list(visits_np.shape) if self._arrays_boundary else []
                ),
                "lightzero_mcts_arrays_boundary_searched_value_shape": (
                    list(searched_values_np.shape) if self._arrays_boundary else []
                ),
                "lightzero_mcts_arrays_boundary_predicted_value_shape": (
                    list(predicted_values_np.shape) if self._arrays_boundary else []
                ),
                "lightzero_mcts_arrays_boundary_policy_logits_shape": (
                    list(policy_logits_np.shape) if self._arrays_boundary else []
                ),
                "lightzero_mcts_arrays_boundary_visit_present_count": (
                    float(visit_present_count) if self._arrays_boundary else 0.0
                ),
                "lightzero_mcts_arrays_boundary_root_value_count": (
                    float(len(root_values)) if self._arrays_boundary else 0.0
                ),
            }
        )

    def run_compact_batch(self, batch: HybridCompactBatch) -> HybridBatchedStackProbeResult:
        self._last_compact_service_root_batch = None
        self._last_compact_service_search_result = None
        self._last_compact_service_replay_chunk = None
        observation, action_mask, compact_telemetry = self._compact_batch_inputs(batch)
        result = self.run(observation, action_mask)
        telemetry = dict(result.telemetry)
        telemetry.update(compact_telemetry)
        if (
            self._arrays_boundary
            and self._arrays_boundary_impl
            in {
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT,
            }
            and self._last_direct_mcts_arrays is not None
        ):
            root_batch, search_result, service_telemetry = (
                self._validate_compact_service_root_result(batch)
            )
            self._last_compact_service_root_batch = root_batch
            self._last_compact_service_search_result = search_result
            telemetry.update(service_telemetry)
        return HybridBatchedStackProbeResult(telemetry=telemetry)

    def run_compact_batch_with_replay_chunk(
        self,
        batch: HybridCompactBatch,
        *,
        chunk: Any,
        record_index: int,
        policy_source: str,
    ) -> HybridBatchedStackProbeResult:
        """Profile-only proof edge from direct compact search output to replay rows."""

        import numpy as np

        result = self.run_compact_batch(batch)
        if (
            self._last_compact_service_root_batch is None
            or self._last_compact_service_search_result is None
        ):
            raise ValueError(
                "compact replay proof requires direct CTree compact service validation"
            )
        started = time.perf_counter()
        replay_chunk = build_compact_replay_chunk_v1_from_search_result(
            chunk,
            batch,
            self._last_compact_service_root_batch,
            self._last_compact_service_search_result,
            record_index=int(record_index),
            policy_source=str(policy_source),
            metadata={
                "profile_backend": self.backend_name,
                "profile_semantics": self.semantics,
            },
        )
        self._last_compact_service_replay_chunk = replay_chunk
        telemetry = dict(result.telemetry)
        telemetry.update(
            {
                "compact_service_replay_chunk_v1_enabled": True,
                "compact_service_replay_chunk_v1_validation_sec": (time.perf_counter() - started),
                "compact_service_replay_chunk_schema_id": str(
                    replay_chunk.metadata.get("schema_id")
                ),
                "compact_service_replay_chunk_record_index": float(replay_chunk.record_index),
                "compact_service_replay_chunk_next_record_index": float(
                    replay_chunk.next_record_index
                ),
                "compact_service_replay_chunk_target_row_count": float(
                    replay_chunk.target_rows.action.size
                ),
                "compact_service_replay_chunk_action_checksum": float(
                    np.asarray(replay_chunk.target_rows.action, dtype=np.int64).sum()
                ),
                "compact_service_replay_chunk_reward_checksum": float(
                    np.asarray(replay_chunk.target_rows.reward, dtype=np.float64).sum()
                ),
            }
        )
        return HybridBatchedStackProbeResult(telemetry=telemetry)

    def _validate_compact_service_root_result(
        self,
        batch: HybridCompactBatch,
    ) -> tuple[Any, Any, dict[str, Any]]:
        import numpy as np

        if self._last_direct_mcts_arrays is None:
            raise ValueError("compact service validation requires direct MCTS arrays")
        started = time.perf_counter()
        root_batch = build_compact_root_batch_v1(
            batch,
            search_lane=self._arrays_boundary_impl,
            metadata={
                "profile_backend": self.backend_name,
                "profile_semantics": self.semantics,
            },
            observation_source=str(
                getattr(batch, "observation_source", COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1)
            ),
            resident_observation=getattr(batch, "resident_observation", None),
        )
        search_arrays = self._last_direct_mcts_arrays
        search_result = compact_search_result_v1_from_arrays(
            root_batch,
            search_arrays,
            default_search_impl=self._arrays_boundary_impl,
            default_num_simulations=self._num_simulations,
            metadata={
                "profile_backend": self.backend_name,
                "profile_semantics": self.semantics,
            },
        )
        telemetry = {
            "compact_service_contract_v1_enabled": True,
            "compact_service_contract_v1_validation_sec": time.perf_counter() - started,
            "compact_service_contract_v1_contract_id": str(root_batch.metadata.get("contract_id")),
            "compact_service_root_batch_schema_id": str(root_batch.metadata.get("schema_id")),
            "compact_service_search_result_schema_id": str(search_result.metadata.get("schema_id")),
            "compact_service_root_count": float(root_batch.legal_mask.shape[0]),
            "compact_service_active_root_count": float(search_result.root_index.size),
            "compact_service_selected_action_checksum": float(
                np.asarray(search_result.selected_action, dtype=np.int64).sum()
            ),
            "compact_service_visit_policy_checksum": float(
                np.asarray(search_result.visit_policy, dtype=np.float64).sum()
            ),
            "compact_service_identity_checksum": float(
                np.asarray(search_result.env_row, dtype=np.int64).sum()
                + np.asarray(search_result.player, dtype=np.int64).sum()
            ),
        }
        return root_batch, search_result, telemetry

    def _compact_batch_inputs(self, batch: HybridCompactBatch) -> tuple[Any, Any, dict[str, Any]]:
        import numpy as np

        stack = np.asarray(batch.observation)
        stack_shape = tuple(int(dim) for dim in stack.shape)
        if len(stack_shape) != 5:
            raise ValueError(
                "LightZero compact batch probe expects observation shape "
                f"[B,P,{POLICY_FRAME_STACK_DEPTH},{TARGET_SIZE},{TARGET_SIZE}], "
                f"got {stack_shape}"
            )
        if stack_shape[2:] != (POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE):
            raise ValueError(
                f"LightZero compact batch probe observation tail mismatch; got {stack_shape[2:]}"
            )
        mask_raw = np.asarray(batch.action_mask, dtype=np.float32)
        expected_mask_shape = stack_shape[:2] + (ACTION_COUNT,)
        if tuple(int(dim) for dim in mask_raw.shape) != expected_mask_shape:
            raise ValueError(
                "LightZero compact batch probe action_mask shape mismatch; "
                f"got {mask_raw.shape}, expected {expected_mask_shape}"
            )
        if not bool(np.all((mask_raw == 0.0) | (mask_raw == 1.0))):
            raise ValueError(
                "LightZero compact batch probe action_mask must be binary 0/1 "
                "to match stock LightZero legal-action semantics"
            )
        mask = mask_raw.astype(np.bool_, copy=False)
        total_root_count = int(stack_shape[0] * stack_shape[1])
        policy_env_id = np.asarray(batch.policy_env_id, dtype=np.int32).reshape(-1)
        policy_env_row = np.asarray(batch.policy_env_row, dtype=np.int32).reshape(-1)
        policy_player = np.asarray(batch.policy_player, dtype=np.int32).reshape(-1)
        if policy_env_id.shape != (total_root_count,):
            raise ValueError("compact batch policy_env_id must have one row per root")
        if policy_env_row.shape != (total_root_count,):
            raise ValueError("compact batch policy_env_row must have one row per root")
        if policy_player.shape != (total_root_count,):
            raise ValueError("compact batch policy_player must have one row per root")
        expected_rows = np.repeat(np.arange(stack_shape[0], dtype=np.int32), stack_shape[1])
        expected_players = np.tile(np.arange(stack_shape[1], dtype=np.int32), stack_shape[0])
        expected_env_ids = (
            expected_rows.astype(np.int64) * int(stack_shape[1]) + expected_players.astype(np.int64)
        ).astype(np.int32, copy=False)
        if not np.array_equal(policy_env_row, expected_rows):
            raise ValueError("compact batch policy_env_row must be row-major for this probe")
        if not np.array_equal(policy_player, expected_players):
            raise ValueError("compact batch policy_player must be row-major for this probe")
        if not np.array_equal(policy_env_id, expected_env_ids):
            raise ValueError("compact batch policy_env_id must be row * player_count + player")

        done = np.asarray(batch.done, dtype=np.bool_)
        if done.shape != stack_shape[:1]:
            raise ValueError("compact batch done must have shape [B]")
        flat_mask = mask.reshape((total_root_count, ACTION_COUNT))
        done_root = np.asarray(batch.done_root, dtype=np.bool_).reshape(-1)
        if done_root.shape != (total_root_count,):
            raise ValueError("compact batch done_root must have one row per root")
        expected_done_root = np.repeat(done, int(stack_shape[1]))
        if not np.array_equal(done_root, expected_done_root):
            raise ValueError("compact batch done_root must equal repeat(done, player_count)")
        active_root_mask = np.asarray(batch.active_root_mask, dtype=np.bool_).reshape(-1)
        if active_root_mask.shape != (total_root_count,):
            raise ValueError("compact batch active_root_mask must have one row per root")
        expected_active_root_mask = np.logical_and(~done_root, flat_mask.any(axis=1))
        if not np.array_equal(active_root_mask, expected_active_root_mask):
            raise ValueError(
                "compact batch active_root_mask must equal action_mask.any(-1) & ~done_root"
            )

        to_play = np.asarray(batch.to_play, dtype=np.int64).reshape(-1)
        if to_play.shape != (total_root_count,):
            raise ValueError("compact batch to_play must have one row per root")
        if bool((to_play[active_root_mask] != self.fixed_opponent_to_play).any()):
            raise ValueError(
                "LightZero compact batch probe currently supports fixed-opponent "
                "to_play=-1 active roots only"
            )

        target_reward = np.asarray(batch.target_reward, dtype=np.float32)
        if target_reward.shape != (total_root_count, 1):
            raise ValueError("compact batch target_reward must have shape [B*P, 1]")
        reward = np.asarray(batch.reward, dtype=np.float32)
        if reward.shape != stack_shape[:2]:
            raise ValueError("compact batch reward must have shape [B, P]")
        if not np.allclose(target_reward, reward.reshape(total_root_count, 1)):
            raise ValueError("compact batch target_reward must match row-major reward")
        for name, rows, row_mask in (
            ("terminal", batch.terminal_global_rows, batch.terminal_row_mask),
            ("autoreset", batch.autoreset_global_rows, batch.autoreset_row_mask),
        ):
            row_values = np.asarray(rows, dtype=np.int32).reshape(-1)
            if row_values.size and (
                bool((row_values < 0).any()) or bool((row_values >= stack_shape[0]).any())
            ):
                raise ValueError(f"compact batch {name}_global_rows are out of range")
            expected_row_mask = np.zeros(stack_shape[0], dtype=np.bool_)
            expected_row_mask[row_values] = True
            actual_row_mask = np.asarray(row_mask, dtype=np.bool_)
            if actual_row_mask.shape != stack_shape[:1]:
                raise ValueError(f"compact batch {name}_row_mask must have shape [B]")
            if not np.array_equal(actual_row_mask, expected_row_mask):
                raise ValueError(f"compact batch {name}_row_mask must match {name}_global_rows")
        final_observation_row_mask = np.asarray(batch.final_observation_row_mask, dtype=np.bool_)
        if final_observation_row_mask.shape != stack_shape[:1]:
            raise ValueError("compact batch final_observation_row_mask must have shape [B]")
        if batch.final_observation is None:
            if bool(final_observation_row_mask.any()):
                raise ValueError(
                    "compact batch final_observation_row_mask must be empty when "
                    "final_observation is absent"
                )
        else:
            final_observation = np.asarray(batch.final_observation)
            if tuple(int(dim) for dim in final_observation.shape) != stack_shape:
                raise ValueError("compact batch final_observation must match observation shape")
            if not np.array_equal(final_observation_row_mask, done):
                raise ValueError("compact batch final_observation_row_mask must match done rows")
        for name, value, expected_shape in (
            ("episode_step", batch.episode_step, stack_shape[:1]),
            ("elapsed_ms", batch.elapsed_ms, stack_shape[:1]),
            ("round_id", batch.round_id, stack_shape[:1]),
            ("alive", batch.alive, stack_shape[:2]),
            ("joint_action", batch.joint_action, stack_shape[:2]),
        ):
            if tuple(int(dim) for dim in np.asarray(value).shape) != expected_shape:
                raise ValueError(f"compact batch {name} must have shape {expected_shape}")

        filtered_mask = flat_mask.copy()
        filtered_mask[~active_root_mask] = False
        return (
            stack,
            filtered_mask.reshape(mask.shape),
            {
                "compact_batch_contract": "compact_row_player_sidecar_v1",
                "lightzero_compact_batch_contract": "compact_row_player_sidecar_v1",
                "lightzero_compact_batch_root_count": float(total_root_count),
                "lightzero_compact_batch_active_root_count": float(active_root_mask.sum()),
                "lightzero_compact_batch_done_root_count": float(done_root.sum()),
                "lightzero_compact_batch_zero_mask_root_count": float(
                    (~flat_mask.any(axis=1)).sum()
                ),
                "lightzero_compact_batch_terminal_count": float(
                    np.asarray(batch.terminal_global_rows, dtype=np.int32).size
                ),
                "lightzero_compact_batch_autoreset_count": float(
                    np.asarray(batch.autoreset_global_rows, dtype=np.int32).size
                ),
                "lightzero_compact_batch_final_observation_present": (
                    batch.final_observation is not None
                ),
                "lightzero_compact_batch_final_observation_rows": float(
                    np.asarray(batch.final_observation_row_mask, dtype=np.bool_).sum()
                ),
                "lightzero_compact_batch_policy_env_id_checksum": float(
                    policy_env_id.astype(np.int64).sum()
                ),
                "lightzero_compact_batch_policy_player_checksum": float(
                    policy_player.astype(np.int64).sum()
                ),
                "lightzero_compact_batch_to_play_checksum": float(to_play.sum()),
                "lightzero_compact_batch_target_reward_checksum": float(
                    target_reward.sum(dtype=np.float64)
                ),
                "lightzero_compact_batch_active_mask_checksum": float(
                    active_root_mask.astype(np.int64).sum()
                ),
            },
        )

    def _run_direct_mcts_arrays(
        self,
        *,
        np: Any,
        torch: Any,
        stack: Any,
        mask: Any,
        flat_stack: Any,
        flat_mask: Any,
        obs_tensor: Any,
        input_telemetry: Mapping[str, Any],
        tensor_prepare_sec: float,
        h2d_sec: float,
        normalize_sec: float,
        total_root_count: int,
        active_root_count: int,
        dropped_zero_mask_root_count: int,
        rows: Any,
        players: Any,
        to_play: list[int],
        device: Any,
        keep_latents_on_device: bool = False,
        precompute_recurrent_outputs: bool = False,
    ) -> HybridBatchedStackProbeResult:
        """Profile-only direct compact arrays boundary over LightZero's real CTree MCTS."""
        from lzero.policy import mz_network_output_unpack, select_action

        model = getattr(self._policy, "_collect_model", None)
        if model is None:
            model = getattr(self._policy, "_model", None)
        mcts = getattr(self._policy, "_mcts_collect", None)
        if model is None or not hasattr(model, "initial_inference"):
            raise ValueError("direct MCTS arrays boundary requires a LightZero collect model")
        if mcts is None or not hasattr(mcts, "search") or not hasattr(type(mcts), "roots"):
            raise ValueError("direct MCTS arrays boundary requires policy._mcts_collect")
        if hasattr(model, "eval"):
            model.eval()

        cfg = getattr(self._policy, "_cfg", SimpleNamespace())
        root_dirichlet_alpha = float(getattr(cfg, "root_dirichlet_alpha", 0.3))
        root_noise_weight = float(getattr(cfg, "root_noise_weight", 0.25))
        eps_cfg = getattr(cfg, "eps", SimpleNamespace(eps_greedy_exploration_in_collect=False))
        eps_greedy = bool(getattr(eps_cfg, "eps_greedy_exploration_in_collect", False))

        started = time.perf_counter()
        with torch.no_grad():
            network_output = model.initial_inference(obs_tensor)
        _sync_torch_device_if_cuda(torch=torch, device=device)
        initial_inference_sec = time.perf_counter() - started

        started = time.perf_counter()
        latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(
            network_output
        )
        unpack_sec = time.perf_counter() - started
        d2h_started = time.perf_counter()
        pred_values_np = (
            self._policy.inverse_scalar_transform_handle(pred_values)
            .detach()
            .cpu()
            .numpy()
            .reshape(active_root_count, -1)[:, 0]
            .astype(np.float32, copy=False)
        )
        if keep_latents_on_device:
            latent_state_roots_np = None
        else:
            latent_state_roots_np = latent_state_roots.detach().cpu().numpy()
        policy_logits_np = policy_logits.detach().cpu().numpy().astype(np.float32, copy=False)
        model_output_d2h_sec = time.perf_counter() - d2h_started
        latent_state_roots_d2h_bytes = (
            0 if latent_state_roots_np is None else int(latent_state_roots_np.nbytes)
        )
        model_output_d2h_bytes = int(
            pred_values_np.nbytes + latent_state_roots_d2h_bytes + policy_logits_np.nbytes
        )
        root_build_started = time.perf_counter()
        policy_logits_list = policy_logits_np.tolist()
        legal_actions = [
            [int(index) for index, value in enumerate(flat_mask[row]) if float(value) == 1.0]
            for row in range(active_root_count)
        ]
        noises = [
            np.random.dirichlet([root_dirichlet_alpha] * len(actions)).astype(np.float32).tolist()
            for actions in legal_actions
        ]
        roots = type(mcts).roots(active_root_count, legal_actions)
        roots.prepare(root_noise_weight, noises, reward_roots, policy_logits_list, to_play)
        root_prepare_sec = unpack_sec + (time.perf_counter() - root_build_started)

        started = time.perf_counter()
        if keep_latents_on_device:
            search_output = self._run_direct_ctree_gpu_latent_search(
                torch=torch,
                np=np,
                mcts=mcts,
                model=model,
                latent_state_roots=latent_state_roots,
                roots=roots,
                to_play=to_play,
                device=device,
                precompute_recurrent_outputs=precompute_recurrent_outputs,
            )
            model_timer_summary = search_output["model_timer_summary"]
            internal_timer_summary = search_output["internal_timer_summary"]
        else:
            model_timer = _LightZeroModelCallTimer(model, device=device)
            internal_timer = _LightZeroCollectForwardInternalTimer(self._policy, device=device)
            with torch.no_grad(), model_timer.patch(), internal_timer.patch():
                mcts.search(roots, model, latent_state_roots_np, to_play)
            _sync_torch_device_if_cuda(torch=torch, device=device)
            model_timer_summary = model_timer.summary()
            internal_timer_summary = internal_timer.summary()
        search_sec = time.perf_counter() - started
        if keep_latents_on_device:
            internal_timer_summary = dict(internal_timer_summary)
            internal_timer_summary["mcts_search_sec"] = float(search_sec)
            internal_timer_summary["mcts_search_calls"] = 1
        gpu_latent_tensor_index_sec = (
            float(search_output.get("tensor_index_sec", 0.0)) if keep_latents_on_device else 0.0
        )
        gpu_latent_leaf_h2d_sec = (
            float(search_output.get("leaf_h2d_sec", 0.0)) if keep_latents_on_device else 0.0
        )
        gpu_latent_output_d2h_sec = (
            float(search_output.get("model_output_d2h_sec", 0.0)) if keep_latents_on_device else 0.0
        )
        gpu_latent_output_d2h_bytes = (
            float(search_output.get("model_output_d2h_bytes", 0.0))
            if keep_latents_on_device
            else 0.0
        )
        gpu_latent_output_listify_sec = (
            float(search_output.get("model_output_listify_sec", 0.0))
            if keep_latents_on_device
            else 0.0
        )
        gpu_latent_precomputed_recurrent_enabled = (
            bool(search_output.get("precomputed_recurrent_outputs_enabled", False))
            if keep_latents_on_device
            else False
        )
        gpu_latent_precomputed_payload_index_sec = (
            float(search_output.get("precomputed_recurrent_payload_index_sec", 0.0))
            if keep_latents_on_device
            else 0.0
        )

        started = time.perf_counter()
        root_visit_counts = roots.get_distributions()
        root_values = roots.get_values()
        actions_np = np.zeros((active_root_count,), dtype=np.int64)
        searched_values_np = np.asarray(root_values, dtype=np.float32).reshape(-1)
        visit_distributions = np.zeros((active_root_count, ACTION_COUNT), dtype=np.float32)
        visit_entropy = np.zeros((active_root_count,), dtype=np.float32)
        illegal_action_count = 0
        all_actions_legal = bool(np.all(flat_mask == 1.0))
        if all_actions_legal:
            legal_visit_matrix = np.asarray(root_visit_counts, dtype=np.float32)
            if tuple(int(dim) for dim in legal_visit_matrix.shape) != (
                active_root_count,
                ACTION_COUNT,
            ):
                raise ValueError(
                    "direct MCTS arrays boundary expected full visit matrix shape "
                    f"{(active_root_count, ACTION_COUNT)}, got {legal_visit_matrix.shape}"
                )
            visit_totals = legal_visit_matrix.sum(axis=1, keepdims=True)
            if bool(np.any(visit_totals <= 0.0)):
                raise ValueError("direct MCTS arrays boundary got zero visit-count row")
            visit_distributions = legal_visit_matrix / visit_totals

            action_scores = legal_visit_matrix.astype(np.float64, copy=False)
            if self._temperature != 1.0:
                action_scores = action_scores ** (1.0 / self._temperature)
            action_score_totals = action_scores.sum(axis=1, keepdims=True)
            if bool(np.any(action_score_totals <= 0.0)):
                raise ValueError("direct MCTS arrays boundary got zero action-probability row")
            action_probs = action_scores / action_score_totals
            if eps_greedy:
                actions_np = np.argmax(legal_visit_matrix, axis=1).astype(np.int64, copy=False)
                if self._epsilon > 0.0:
                    replace_mask = np.random.random(active_root_count) < self._epsilon
                    replace_count = int(np.count_nonzero(replace_mask))
                    if replace_count:
                        actions_np[replace_mask] = np.random.randint(
                            0,
                            ACTION_COUNT,
                            size=replace_count,
                            dtype=np.int64,
                        )
            else:
                cdf = np.cumsum(action_probs, axis=1)
                samples = np.random.random(active_root_count)
                actions_np = np.sum(cdf < samples[:, None], axis=1).astype(
                    np.int64,
                    copy=False,
                )
                actions_np = np.clip(actions_np, 0, ACTION_COUNT - 1)
            safe_action_probs = np.where(action_probs > 0.0, action_probs, 1.0)
            visit_entropy = (
                -np.sum(
                    np.where(
                        action_probs > 0.0,
                        action_probs * np.log2(safe_action_probs),
                        0.0,
                    ),
                    axis=1,
                )
                .astype(np.float32, copy=False)
                .reshape(active_root_count)
            )
        else:
            for row in range(active_root_count):
                legal = legal_actions[row]
                legal_visit = np.asarray(root_visit_counts[row], dtype=np.float32).reshape(-1)
                if legal_visit.size != len(legal):
                    raise ValueError(
                        "direct MCTS arrays boundary got visit count length "
                        f"{legal_visit.size} for {len(legal)} legal actions"
                    )
                if legal_visit.size:
                    visit_distributions[row, np.asarray(legal, dtype=np.int64)] = legal_visit
                total = float(visit_distributions[row].sum())
                if total > 0.0:
                    visit_distributions[row] /= np.float32(total)
                if eps_greedy:
                    action_index, entropy_value = select_action(
                        legal_visit,
                        temperature=self._temperature,
                        deterministic=True,
                    )
                    action = int(legal[int(action_index)])
                    if np.random.rand() < self._epsilon:
                        action = int(np.random.choice(legal))
                else:
                    action_index, entropy_value = select_action(
                        legal_visit,
                        temperature=self._temperature,
                        deterministic=False,
                    )
                    action = int(legal[int(action_index)])
                actions_np[row] = action
                visit_entropy[row] = np.float32(entropy_value)
                if action < 0 or action >= ACTION_COUNT or float(flat_mask[row, action]) != 1.0:
                    illegal_action_count += 1
        if illegal_action_count:
            raise ValueError(
                "direct MCTS arrays boundary decoded illegal actions: "
                f"{illegal_action_count} / {active_root_count}"
            )
        action_checksum = int(
            sum((index + 1) * (int(action) + 1) for index, action in enumerate(actions_np))
        )
        debug_arrays = _small_compact_mcts_arrays_debug(
            np=np,
            actions=actions_np,
            searched_values=searched_values_np,
            predicted_values=pred_values_np,
            policy_logits=policy_logits_np,
            visit_distributions=visit_distributions,
        )
        self._last_direct_mcts_arrays = {
            "selected_action": actions_np.astype(np.int16, copy=True),
            "visit_policy": visit_distributions.astype(np.float32, copy=True),
            "root_value": searched_values_np.astype(np.float32, copy=True),
            "predicted_value": pred_values_np.astype(np.float32, copy=True),
            "predicted_policy_logits": policy_logits_np.astype(np.float32, copy=True),
            "array_source": "direct_mcts_arrays",
            "search_impl": str(self._arrays_boundary_impl),
            "actual_search_simulations": int(self._num_simulations),
            "requested_simulations": int(self._num_simulations),
        }
        compact_output_bytes = int(
            actions_np.nbytes
            + searched_values_np.nbytes
            + pred_values_np.nbytes
            + policy_logits_np.nbytes
            + visit_distributions.nbytes
            + visit_entropy.nbytes
        )
        output_assembly_sec = time.perf_counter() - started
        readback_sec = 0.0
        recurrent_model_sec = float(model_timer_summary["recurrent_inference_sec"])
        recurrent_model_calls = int(model_timer_summary["recurrent_inference_calls"])
        logical_recurrent_eval_count = int(active_root_count * self._num_simulations)
        actual_recurrent_eval_count = int(active_root_count * recurrent_model_calls)
        logical_model_eval_count = int(active_root_count + logical_recurrent_eval_count)
        actual_model_eval_count = int(active_root_count + actual_recurrent_eval_count)
        synthetic_recurrent_eval_count = max(
            0,
            logical_recurrent_eval_count - actual_recurrent_eval_count,
        )
        model_total_sec = initial_inference_sec + recurrent_model_sec
        mcts_search_sec = float(internal_timer_summary["mcts_search_sec"])
        ctree_traverse_sec = float(internal_timer_summary["ctree_batch_traverse_sec"])
        ctree_backprop_sec = float(internal_timer_summary["ctree_batch_backpropagate_sec"])
        input_preprocess_sec = float(input_telemetry.get("input_prepare_sec", 0.0))
        input_transfer_bytes = float(
            input_telemetry.get("transfer_bytes", getattr(flat_stack, "nbytes", 0))
        )
        input_freshness = str(input_telemetry.get("input_freshness", "fresh"))
        total_sec = (
            tensor_prepare_sec
            + input_preprocess_sec
            + h2d_sec
            + normalize_sec
            + initial_inference_sec
            + model_output_d2h_sec
            + root_prepare_sec
            + search_sec
            + output_assembly_sec
            + readback_sec
        )
        direct_boundary_non_model_sec = max(0.0, total_sec - model_total_sec)
        return HybridBatchedStackProbeResult(
            telemetry={
                "host_to_device_sec": h2d_sec,
                "host_to_device_bytes": input_transfer_bytes,
                "input_transfer_bytes": input_transfer_bytes,
                "normalize_sec": normalize_sec,
                "device_sec": initial_inference_sec + search_sec,
                "readback_sec": readback_sec,
                "total_sec": total_sec,
                "simulations": float(self._num_simulations),
                "channels": 0.0,
                "roots": float(active_root_count),
                "input_rank": float(len(getattr(stack, "shape", ()))),
                "input_bytes": float(getattr(stack, "nbytes", 0)),
                "model_eval_count": float(logical_model_eval_count),
                "logical_model_eval_count": float(logical_model_eval_count),
                "actual_model_eval_count": float(actual_model_eval_count),
                "synthetic_recurrent_eval_count": float(synthetic_recurrent_eval_count),
                "output_readback_bytes": float(compact_output_bytes),
                "action_mask_consumed": 1.0,
                "compile_excluded_by_warmup": 1.0,
                "lightzero_consumer_total_sec": total_sec,
                "lightzero_consumer_tensor_prepare_sec": tensor_prepare_sec,
                "lightzero_consumer_h2d_sec": h2d_sec,
                "lightzero_consumer_normalize_sec": normalize_sec,
                "lightzero_consumer_input_mode": self._input_mode,
                "lightzero_consumer_input_freshness": input_freshness,
                "lightzero_consumer_pin_memory_sec": float(
                    input_telemetry.get("pin_memory_sec", 0.0)
                ),
                "lightzero_consumer_host_prenormalize_sec": float(
                    input_telemetry.get("host_prenormalize_sec", 0.0)
                ),
                "lightzero_consumer_input_prepare_sec": input_preprocess_sec,
                "lightzero_consumer_resident_first_fill_sec": float(
                    input_telemetry.get("resident_first_fill_sec", 0.0)
                ),
                "lightzero_consumer_resident_reused": float(
                    input_telemetry.get("resident_reused", 0.0)
                ),
                "lightzero_consumer_collect_forward_sec": 0.0,
                "lightzero_consumer_collect_forward_wall_sec": 0.0,
                "lightzero_consumer_model_initial_inference_sec": initial_inference_sec,
                "lightzero_consumer_model_initial_inference_calls": 1.0,
                "lightzero_consumer_model_recurrent_inference_sec": recurrent_model_sec,
                "lightzero_consumer_model_recurrent_inference_calls": float(recurrent_model_calls),
                "lightzero_consumer_logical_model_eval_count": float(logical_model_eval_count),
                "lightzero_consumer_actual_model_eval_count": float(actual_model_eval_count),
                "lightzero_consumer_synthetic_recurrent_eval_count": float(
                    synthetic_recurrent_eval_count
                ),
                "lightzero_consumer_model_total_sec": model_total_sec,
                "lightzero_consumer_model_output_d2h_sec": model_output_d2h_sec,
                "lightzero_consumer_model_output_d2h_bytes": float(model_output_d2h_bytes),
                "lightzero_consumer_gpu_latent_enabled": bool(keep_latents_on_device),
                "lightzero_consumer_gpu_latent_tensor_index_sec": (gpu_latent_tensor_index_sec),
                "lightzero_consumer_gpu_latent_leaf_h2d_sec": gpu_latent_leaf_h2d_sec,
                "lightzero_consumer_gpu_latent_search_output_d2h_sec": (gpu_latent_output_d2h_sec),
                "lightzero_consumer_gpu_latent_search_output_d2h_bytes": (
                    gpu_latent_output_d2h_bytes
                ),
                "lightzero_consumer_gpu_latent_search_output_listify_sec": (
                    gpu_latent_output_listify_sec
                ),
                "lightzero_consumer_gpu_latent_precomputed_recurrent_enabled": (
                    bool(gpu_latent_precomputed_recurrent_enabled)
                ),
                "lightzero_consumer_gpu_latent_precomputed_payload_index_sec": (
                    gpu_latent_precomputed_payload_index_sec
                ),
                "lightzero_consumer_collect_forward_non_model_sec": 0.0,
                "lightzero_consumer_direct_boundary_non_model_sec": (direct_boundary_non_model_sec),
                "lightzero_consumer_model_timer_status": str(model_timer_summary["status"]),
                "lightzero_consumer_mcts_timer_status": str(internal_timer_summary["status"]),
                "lightzero_consumer_mcts_timer_patched_metric_count": float(
                    internal_timer_summary["patched_metric_count"]
                ),
                "lightzero_consumer_mcts_timer_error_count": float(
                    len(internal_timer_summary["errors"])
                ),
                "lightzero_consumer_mcts_timer_errors": internal_timer_summary["errors"],
                "lightzero_consumer_mcts_search_sec": mcts_search_sec,
                "lightzero_consumer_mcts_search_calls": float(
                    internal_timer_summary["mcts_search_calls"]
                ),
                "lightzero_consumer_mcts_search_non_model_sec": max(
                    0.0, mcts_search_sec - recurrent_model_sec
                ),
                "lightzero_consumer_collect_forward_outside_mcts_sec": 0.0,
                "lightzero_consumer_collect_forward_outside_mcts_non_initial_model_sec": (0.0),
                "lightzero_consumer_ctree_batch_traverse_sec": ctree_traverse_sec,
                "lightzero_consumer_ctree_batch_traverse_calls": float(
                    internal_timer_summary["ctree_batch_traverse_calls"]
                ),
                "lightzero_consumer_ctree_batch_backpropagate_sec": ctree_backprop_sec,
                "lightzero_consumer_ctree_batch_backpropagate_calls": float(
                    internal_timer_summary["ctree_batch_backpropagate_calls"]
                ),
                "lightzero_consumer_decode_sec": output_assembly_sec,
                "lightzero_consumer_readback_sec": readback_sec,
                "lightzero_consumer_stack_h2d_bytes": input_transfer_bytes,
                "lightzero_consumer_input_transfer_bytes": input_transfer_bytes,
                "lightzero_consumer_mask_numpy_bytes": float(getattr(flat_mask, "nbytes", 0)),
                "lightzero_consumer_input_bytes": float(getattr(stack, "nbytes", 0))
                + float(getattr(mask, "nbytes", 0)),
                "lightzero_consumer_output_bytes": float(compact_output_bytes),
                "lightzero_consumer_policy_device": str(device),
                "lightzero_consumer_policy_class": str(
                    self._policy_metadata.get("policy_class", type(self._policy).__name__)
                ),
                "lightzero_consumer_policy_surface": dict(self._policy_metadata.get("surface", {})),
                "lightzero_consumer_num_simulations": float(self._num_simulations),
                "lightzero_consumer_collect_with_pure_policy": False,
                "lightzero_consumer_cpu_tree_included": 1.0,
                "lightzero_policy_forward_calls": 0.0,
                "lightzero_total_root_count": float(total_root_count),
                "lightzero_filtered_zero_mask_root_count": float(dropped_zero_mask_root_count),
                "lightzero_root_count": float(active_root_count),
                "lightzero_roots_per_call": float(active_root_count),
                "lightzero_ready_env_id_first": 0,
                "lightzero_ready_env_id_last": int(active_root_count - 1),
                "lightzero_to_play_mode": "fixed_opponent_minus_one",
                "lightzero_to_play_sample": to_play[: min(8, len(to_play))],
                "lightzero_output_key_sample": ["compact_arrays"],
                "lightzero_action_checksum": float(action_checksum),
                "lightzero_illegal_action_count": float(illegal_action_count),
                "lightzero_visit_distribution_count": float(active_root_count),
                "lightzero_root_value_count": float(active_root_count),
                "lightzero_root_value_mean": (
                    float(searched_values_np.mean()) if searched_values_np.size else 0.0
                ),
                "lightzero_first_actions": actions_np[: min(16, len(actions_np))]
                .astype(int)
                .tolist(),
                "lightzero_rows_sample": rows[: min(8, len(rows))].astype(int).tolist(),
                "lightzero_players_sample": players[: min(8, len(players))].astype(int).tolist(),
                "lightzero_policy_build_sec": float(self._policy_metadata.get("build_sec", 0.0)),
                "lightzero_policy_requested_cuda": bool(
                    self._policy_metadata.get("requested_cuda", False)
                ),
                "lightzero_policy_cuda": bool(self._policy_metadata.get("cuda", False)),
                "lightzero_mcts_arrays_boundary_enabled": True,
                "lightzero_mcts_arrays_boundary_semantics": self.semantics,
                "lightzero_mcts_arrays_boundary_impl": self._arrays_boundary_impl,
                "lightzero_mcts_arrays_boundary_input_mode": self._input_mode,
                "lightzero_mcts_arrays_boundary_input_freshness": input_freshness,
                "lightzero_mcts_arrays_boundary_pin_memory_sec": float(
                    input_telemetry.get("pin_memory_sec", 0.0)
                ),
                "lightzero_mcts_arrays_boundary_host_prenormalize_sec": float(
                    input_telemetry.get("host_prenormalize_sec", 0.0)
                ),
                "lightzero_mcts_arrays_boundary_input_prepare_sec": (input_preprocess_sec),
                "lightzero_mcts_arrays_boundary_input_transfer_bytes": (input_transfer_bytes),
                "lightzero_mcts_arrays_boundary_obs_h2d_bytes": input_transfer_bytes,
                "lightzero_mcts_arrays_boundary_mask_h2d_bytes": 0.0,
                "lightzero_mcts_arrays_boundary_action_d2h_bytes": 0.0,
                "lightzero_mcts_arrays_boundary_replay_payload_d2h_bytes": 0.0,
                "lightzero_mcts_arrays_boundary_root_observation_copy_bytes": 0.0,
                "lightzero_mcts_arrays_boundary_python_rows_materialized": 0.0,
                "lightzero_mcts_arrays_boundary_rnd_materialized_rows": 0.0,
                "lightzero_mcts_arrays_boundary_resident_first_fill_sec": float(
                    input_telemetry.get("resident_first_fill_sec", 0.0)
                ),
                "lightzero_mcts_arrays_boundary_resident_reused": float(
                    input_telemetry.get("resident_reused", 0.0)
                ),
                "lightzero_mcts_arrays_boundary_total_sec": total_sec,
                "lightzero_mcts_arrays_boundary_collect_forward_sec": 0.0,
                "lightzero_mcts_arrays_boundary_decode_sec": output_assembly_sec,
                "lightzero_mcts_arrays_boundary_initial_inference_sec": (initial_inference_sec),
                "lightzero_mcts_arrays_boundary_logical_model_eval_count": float(
                    logical_model_eval_count
                ),
                "lightzero_mcts_arrays_boundary_actual_model_eval_count": float(
                    actual_model_eval_count
                ),
                "lightzero_mcts_arrays_boundary_synthetic_recurrent_eval_count": float(
                    synthetic_recurrent_eval_count
                ),
                "lightzero_mcts_arrays_boundary_model_output_d2h_sec": (model_output_d2h_sec),
                "lightzero_mcts_arrays_boundary_model_output_d2h_bytes": float(
                    model_output_d2h_bytes
                ),
                "lightzero_mcts_arrays_boundary_gpu_latent_enabled": bool(keep_latents_on_device),
                "lightzero_mcts_arrays_boundary_gpu_latent_tensor_index_sec": (
                    gpu_latent_tensor_index_sec
                ),
                "lightzero_mcts_arrays_boundary_gpu_latent_leaf_h2d_sec": (gpu_latent_leaf_h2d_sec),
                "lightzero_mcts_arrays_boundary_gpu_latent_search_output_d2h_sec": (
                    gpu_latent_output_d2h_sec
                ),
                "lightzero_mcts_arrays_boundary_gpu_latent_search_output_d2h_bytes": (
                    gpu_latent_output_d2h_bytes
                ),
                "lightzero_mcts_arrays_boundary_gpu_latent_search_output_listify_sec": (
                    gpu_latent_output_listify_sec
                ),
                "lightzero_mcts_arrays_boundary_gpu_latent_precomputed_recurrent_enabled": (
                    bool(gpu_latent_precomputed_recurrent_enabled)
                ),
                "lightzero_mcts_arrays_boundary_gpu_latent_precomputed_payload_index_sec": (
                    gpu_latent_precomputed_payload_index_sec
                ),
                "lightzero_mcts_arrays_boundary_root_prepare_sec": root_prepare_sec,
                "lightzero_mcts_arrays_boundary_non_model_sec": (direct_boundary_non_model_sec),
                "lightzero_mcts_arrays_boundary_search_sec": search_sec,
                "lightzero_mcts_arrays_boundary_output_assembly_sec": (output_assembly_sec),
                "lightzero_mcts_arrays_boundary_compact_output_bytes": float(compact_output_bytes),
                "lightzero_mcts_arrays_boundary_public_output_bytes": 0.0,
                "lightzero_mcts_arrays_boundary_action_shape": list(actions_np.shape),
                "lightzero_mcts_arrays_boundary_debug_arrays": debug_arrays,
                "lightzero_mcts_arrays_boundary_visit_shape": list(visit_distributions.shape),
                "lightzero_mcts_arrays_boundary_searched_value_shape": list(
                    searched_values_np.shape
                ),
                "lightzero_mcts_arrays_boundary_predicted_value_shape": list(pred_values_np.shape),
                "lightzero_mcts_arrays_boundary_policy_logits_shape": list(policy_logits_np.shape),
                "lightzero_mcts_arrays_boundary_all_actions_legal_fast_path": bool(
                    all_actions_legal
                ),
                "lightzero_mcts_arrays_boundary_visit_present_count": float(active_root_count),
                "lightzero_mcts_arrays_boundary_root_value_count": float(active_root_count),
            }
        )

    def _run_direct_ctree_gpu_latent_search(
        self,
        *,
        torch: Any,
        np: Any,
        mcts: Any,
        model: Any,
        latent_state_roots: Any,
        roots: Any,
        to_play: list[int],
        device: Any,
        precompute_recurrent_outputs: bool = False,
    ) -> dict[str, Any]:
        """Run LightZero CTree search while keeping latent-state storage on device.

        CTree itself still needs CPU policy/reward/value arrays for node
        expansion and backup. This prototype only removes the repeated
        leaf-latent CPU round trip inside LightZero's Python search loop.
        """
        from lzero.policy import InverseScalarTransform, mz_network_output_unpack

        import lzero.mcts.tree_search.mcts_ctree as mcts_ctree

        tree_muzero = mcts_ctree.tree_muzero
        batch_size = int(roots.num)
        cfg = getattr(mcts, "_cfg", getattr(self._policy, "_cfg", SimpleNamespace()))
        pb_c_base = int(getattr(cfg, "pb_c_base"))
        pb_c_init = float(getattr(cfg, "pb_c_init"))
        discount_factor = float(getattr(cfg, "discount_factor"))
        num_simulations = int(getattr(cfg, "num_simulations", self._num_simulations))
        value_delta_max = float(getattr(cfg, "value_delta_max", 0.01))
        support_scale = getattr(getattr(cfg, "model", SimpleNamespace()), "support_scale")
        categorical_distribution = getattr(
            getattr(cfg, "model", SimpleNamespace()),
            "categorical_distribution",
        )
        inverse_scalar_transform = InverseScalarTransform(
            support_scale,
            getattr(cfg, "device", device),
            categorical_distribution,
        )
        latent_pool = torch.empty(
            (num_simulations + 1, batch_size)
            + tuple(int(dim) for dim in latent_state_roots.shape[1:]),
            dtype=latent_state_roots.dtype,
            device=device,
        )
        latent_pool[0].copy_(latent_state_roots)
        min_max_stats_lst = tree_muzero.MinMaxStatsList(batch_size)
        min_max_stats_lst.set_delta(value_delta_max)

        timers = {
            "recurrent_inference": 0.0,
            "ctree_batch_traverse": 0.0,
            "ctree_batch_backpropagate": 0.0,
            "tensor_index": 0.0,
            "leaf_h2d": 0.0,
            "model_output_d2h": 0.0,
            "model_output_listify": 0.0,
            "precomputed_recurrent_payload_index": 0.0,
        }
        calls = {
            "recurrent_inference": 0,
            "ctree_batch_traverse": 0,
            "ctree_batch_backpropagate": 0,
        }
        model_output_d2h_bytes = 0
        to_play_batch = list(to_play)
        precomputed_payload = None
        if precompute_recurrent_outputs:
            base_logits = torch.zeros(
                (batch_size, ACTION_COUNT),
                dtype=torch.float32,
                device=device,
            )
            precomputed_payload = {
                "reward": torch.zeros((batch_size, 1), dtype=torch.float32, device=device),
                "value": torch.zeros((batch_size, 1), dtype=torch.float32, device=device),
                "policy_logits": base_logits,
            }

        with torch.no_grad():
            model.eval()
            for simulation_index in range(num_simulations):
                results = tree_muzero.ResultsWrapper(num=batch_size)
                started = time.perf_counter()
                (
                    latent_state_index_in_search_path,
                    latent_state_index_in_batch,
                    last_actions,
                    virtual_to_play_batch,
                ) = tree_muzero.batch_traverse(
                    roots,
                    pb_c_base,
                    pb_c_init,
                    discount_factor,
                    min_max_stats_lst,
                    results,
                    to_play_batch,
                )
                timers["ctree_batch_traverse"] += time.perf_counter() - started
                calls["ctree_batch_traverse"] += 1

                started = time.perf_counter()
                if not latent_state_index_in_search_path:
                    raise ValueError("direct_ctree_gpu_latent got no latent indices")
                path_indices = torch.as_tensor(
                    latent_state_index_in_search_path,
                    dtype=torch.long,
                    device=device,
                )
                batch_indices = torch.as_tensor(
                    latent_state_index_in_batch,
                    dtype=torch.long,
                    device=device,
                )
                latent_states = latent_pool[path_indices, batch_indices]
                timers["tensor_index"] += time.perf_counter() - started

                started = time.perf_counter()
                last_actions_tensor = torch.as_tensor(
                    np.asarray(last_actions),
                    dtype=torch.long,
                    device=device,
                )
                _sync_torch_device_if_cuda(torch=torch, device=device)
                timers["leaf_h2d"] += time.perf_counter() - started

                if precomputed_payload is None:
                    started = time.perf_counter()
                    network_output = model.recurrent_inference(latent_states, last_actions_tensor)
                    _sync_torch_device_if_cuda(torch=torch, device=device)
                    timers["recurrent_inference"] += time.perf_counter() - started
                    calls["recurrent_inference"] += 1

                    started = time.perf_counter()
                    next_latent_state, reward, value, policy_logits = mz_network_output_unpack(
                        network_output
                    )
                    reward_plain = inverse_scalar_transform(reward).reshape(batch_size, 1)
                    value_plain = inverse_scalar_transform(value).reshape(batch_size, 1)
                    policy_logits_plain = policy_logits.to(dtype=torch.float32).reshape(
                        batch_size,
                        -1,
                    )
                else:
                    started = time.perf_counter()
                    next_latent_state = latent_states
                    reward_plain = precomputed_payload["reward"]
                    value_plain = precomputed_payload["value"]
                    policy_logits_plain = precomputed_payload["policy_logits"]
                    timers["precomputed_recurrent_payload_index"] += time.perf_counter() - started

                started = time.perf_counter()
                model_output_np = (
                    torch.cat(
                        (reward_plain, value_plain, policy_logits_plain),
                        dim=1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                reward_np = model_output_np[:, 0]
                value_np = model_output_np[:, 1]
                policy_logits_np = model_output_np[:, 2:]
                _sync_torch_device_if_cuda(torch=torch, device=device)
                timers["model_output_d2h"] += time.perf_counter() - started
                model_output_d2h_bytes += int(model_output_np.nbytes)
                latent_pool[simulation_index + 1].copy_(next_latent_state)

                started = time.perf_counter()
                reward_batch = reward_np.reshape(-1).tolist()
                value_batch = value_np.reshape(-1).tolist()
                policy_logits_batch = policy_logits_np.tolist()
                timers["model_output_listify"] += time.perf_counter() - started
                started = time.perf_counter()
                tree_muzero.batch_backpropagate(
                    simulation_index + 1,
                    discount_factor,
                    reward_batch,
                    value_batch,
                    policy_logits_batch,
                    min_max_stats_lst,
                    results,
                    virtual_to_play_batch,
                )
                timers["ctree_batch_backpropagate"] += time.perf_counter() - started
                calls["ctree_batch_backpropagate"] += 1

        timer_status = (
            "installed_gpu_latent_precomputed_recurrent"
            if precompute_recurrent_outputs
            else "installed_gpu_latent"
        )
        return {
            "model_timer_summary": {
                "status": timer_status,
                "initial_inference_sec": 0.0,
                "initial_inference_calls": 0,
                "recurrent_inference_sec": float(timers["recurrent_inference"]),
                "recurrent_inference_calls": int(calls["recurrent_inference"]),
                "model_total_sec": float(timers["recurrent_inference"]),
            },
            "internal_timer_summary": {
                "status": timer_status,
                "patched_metric_count": 2,
                "errors": [],
                "mcts_search_sec": 0.0,
                "mcts_search_calls": 0,
                "ctree_batch_traverse_sec": float(timers["ctree_batch_traverse"]),
                "ctree_batch_traverse_calls": int(calls["ctree_batch_traverse"]),
                "ctree_batch_backpropagate_sec": float(timers["ctree_batch_backpropagate"]),
                "ctree_batch_backpropagate_calls": int(calls["ctree_batch_backpropagate"]),
            },
            "tensor_index_sec": float(timers["tensor_index"]),
            "leaf_h2d_sec": float(timers["leaf_h2d"]),
            "model_output_d2h_sec": float(timers["model_output_d2h"]),
            "model_output_d2h_bytes": float(model_output_d2h_bytes),
            "model_output_listify_sec": float(timers["model_output_listify"]),
            "precomputed_recurrent_outputs_enabled": bool(precompute_recurrent_outputs),
            "precomputed_recurrent_payload_index_sec": float(
                timers["precomputed_recurrent_payload_index"]
            ),
        }

    def _prepare_observation_tensor(
        self,
        *,
        np: Any,
        torch: Any,
        flat_stack: Any,
        stack_dtype: Any,
        device: Any,
    ) -> tuple[Any, float, float, dict[str, Any]]:
        telemetry = {
            "pin_memory_sec": 0.0,
            "host_prenormalize_sec": 0.0,
            "input_prepare_sec": 0.0,
            "resident_first_fill_sec": 0.0,
            "resident_reused": 0.0,
            "transfer_bytes": float(getattr(flat_stack, "nbytes", 0)),
            "input_freshness": "fresh",
        }

        if self._input_mode == LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_FLOAT32:
            started = time.perf_counter()
            if stack_dtype == np.uint8:
                prepared_stack = flat_stack.astype(np.float32, copy=False) * np.float32(1.0 / 255.0)
            else:
                prepared_stack = flat_stack.astype(np.float32, copy=False)
            telemetry["host_prenormalize_sec"] = time.perf_counter() - started
            telemetry["input_prepare_sec"] = telemetry["host_prenormalize_sec"]
            telemetry["transfer_bytes"] = float(getattr(prepared_stack, "nbytes", 0))

            started = time.perf_counter()
            obs_tensor = torch.as_tensor(prepared_stack, dtype=torch.float32, device=device)
            _sync_torch_device_if_cuda(torch=torch, device=device)
            h2d_sec = time.perf_counter() - started
            return obs_tensor, h2d_sec, 0.0, telemetry

        if self._input_mode == LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8_PINNED:
            started = time.perf_counter()
            cpu_dtype = getattr(torch, "uint8", None) if stack_dtype == np.uint8 else torch.float32
            if cpu_dtype is None:
                cpu_tensor = torch.as_tensor(flat_stack)
            else:
                cpu_tensor = torch.as_tensor(flat_stack, dtype=cpu_dtype)
            tensor_wrap_sec = time.perf_counter() - started
            if str(device).startswith("cuda") and hasattr(cpu_tensor, "pin_memory"):
                pin_started = time.perf_counter()
                cpu_tensor = cpu_tensor.pin_memory()
                telemetry["pin_memory_sec"] = time.perf_counter() - pin_started
            telemetry["input_prepare_sec"] = tensor_wrap_sec + float(telemetry["pin_memory_sec"])
            started = time.perf_counter()
            if hasattr(cpu_tensor, "to"):
                obs_tensor = cpu_tensor.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=str(device).startswith("cuda"),
                )
            else:
                obs_tensor = torch.as_tensor(flat_stack, dtype=torch.float32, device=device)
            _sync_torch_device_if_cuda(torch=torch, device=device)
            h2d_sec = time.perf_counter() - started

            started = time.perf_counter()
            normalize_sec = 0.0
            if stack_dtype == np.uint8:
                obs_tensor = obs_tensor * (1.0 / 255.0)
                _sync_torch_device_if_cuda(torch=torch, device=device)
                normalize_sec = time.perf_counter() - started
            return obs_tensor, h2d_sec, normalize_sec, telemetry

        if self._input_mode == LIGHTZERO_ARRAY_CEILING_INPUT_MODE_RESIDENT_TORCH_REUSE:
            signature = (
                tuple(int(dim) for dim in getattr(flat_stack, "shape", ())),
                str(device),
                str(stack_dtype),
            )
            if self._resident_obs_tensor is not None and self._resident_obs_signature == signature:
                telemetry["resident_reused"] = 1.0
                telemetry["input_freshness"] = "stale_profile_ceiling"
                telemetry["transfer_bytes"] = 0.0
                return self._resident_obs_tensor, 0.0, 0.0, telemetry

            started = time.perf_counter()
            obs_tensor = torch.as_tensor(flat_stack, dtype=torch.float32, device=device)
            _sync_torch_device_if_cuda(torch=torch, device=device)
            h2d_sec = time.perf_counter() - started

            started = time.perf_counter()
            normalize_sec = 0.0
            if stack_dtype == np.uint8:
                obs_tensor = obs_tensor * (1.0 / 255.0)
                _sync_torch_device_if_cuda(torch=torch, device=device)
                normalize_sec = time.perf_counter() - started
            self._resident_obs_tensor = obs_tensor
            self._resident_obs_signature = signature
            telemetry["resident_first_fill_sec"] = h2d_sec + normalize_sec
            telemetry["input_freshness"] = "fresh_first_fill"
            return obs_tensor, h2d_sec, normalize_sec, telemetry

        started = time.perf_counter()
        obs_tensor = torch.as_tensor(flat_stack, dtype=torch.float32, device=device)
        _sync_torch_device_if_cuda(torch=torch, device=device)
        h2d_sec = time.perf_counter() - started

        started = time.perf_counter()
        normalize_sec = 0.0
        if stack_dtype == np.uint8:
            obs_tensor = obs_tensor * (1.0 / 255.0)
            _sync_torch_device_if_cuda(torch=torch, device=device)
            normalize_sec = time.perf_counter() - started
        return obs_tensor, h2d_sec, normalize_sec, telemetry


class _LightZeroInitialInferenceStackProbe:
    """Profile-only real MuZero model inference consumer for pre-scalar stacks."""

    backend_name = "lightzero_initial_inference_consumer"
    semantics = "lightzero_model_initial_inference_only"

    def __init__(
        self,
        *,
        policy: Any,
        policy_metadata: Mapping[str, Any],
    ) -> None:
        self._policy = policy
        self._policy_metadata = dict(policy_metadata)

    def run(self, observation: Any, action_mask: Any) -> HybridBatchedStackProbeResult:
        import numpy as np
        import torch

        model = getattr(self._policy, "_model", None)
        if model is None or not hasattr(model, "initial_inference"):
            raise ValueError("LightZero initial-inference probe requires policy._model")

        stack = np.asarray(observation)
        stack_shape = tuple(int(dim) for dim in stack.shape)
        if stack_shape[2:] != (POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE):
            raise ValueError(
                "LightZero initial-inference probe expects observation shape "
                f"[B,P,{POLICY_FRAME_STACK_DEPTH},{TARGET_SIZE},{TARGET_SIZE}], "
                f"got {stack_shape}"
            )
        mask = np.asarray(action_mask, dtype=np.float32)
        expected_mask_shape = stack_shape[:2] + (ACTION_COUNT,)
        if tuple(int(dim) for dim in mask.shape) != expected_mask_shape:
            raise ValueError(
                "LightZero initial-inference probe action_mask shape mismatch; "
                f"got {mask.shape}, expected {expected_mask_shape}"
            )
        total_root_count = int(stack_shape[0] * stack_shape[1])
        flat_mask_all = mask.reshape((total_root_count, ACTION_COUNT))
        if not np.all((flat_mask_all == 0.0) | (flat_mask_all == 1.0)):
            raise ValueError("LightZero initial-inference probe requires binary action masks")
        legal_root_mask = np.any(flat_mask_all > 0.0, axis=1)
        active_root_count = int(np.count_nonzero(legal_root_mask))
        dropped_zero_mask_root_count = int(total_root_count - active_root_count)
        device = _policy_model_device(self._policy)

        started = time.perf_counter()
        flat_stack_all = stack.reshape(
            (total_root_count, POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE)
        )
        flat_stack = flat_stack_all[legal_root_mask]
        tensor_prepare_sec = time.perf_counter() - started

        if active_root_count == 0:
            return HybridBatchedStackProbeResult(
                telemetry={
                    "host_to_device_sec": 0.0,
                    "host_to_device_bytes": 0.0,
                    "normalize_sec": 0.0,
                    "device_sec": 0.0,
                    "readback_sec": 0.0,
                    "total_sec": tensor_prepare_sec,
                    "simulations": 0.0,
                    "channels": 0.0,
                    "roots": 0.0,
                    "input_rank": float(len(stack_shape)),
                    "input_bytes": float(getattr(stack, "nbytes", 0)),
                    "model_eval_count": 0.0,
                    "output_readback_bytes": 0.0,
                    "lightzero_initial_inference_total_sec": tensor_prepare_sec,
                    "lightzero_initial_inference_tensor_prepare_sec": tensor_prepare_sec,
                    "lightzero_initial_inference_h2d_sec": 0.0,
                    "lightzero_initial_inference_normalize_sec": 0.0,
                    "lightzero_initial_inference_forward_sec": 0.0,
                    "lightzero_initial_inference_readback_sec": 0.0,
                    "lightzero_initial_inference_stack_h2d_bytes": 0.0,
                    "lightzero_initial_inference_policy_device": str(device),
                    "lightzero_initial_inference_policy_class": str(
                        self._policy_metadata.get("policy_class", type(self._policy).__name__)
                    ),
                    "lightzero_initial_inference_model_class": str(
                        self._policy_metadata.get("model_class", type(model).__name__)
                    ),
                    "lightzero_initial_inference_policy_surface": dict(
                        self._policy_metadata.get("surface", {})
                    ),
                    "lightzero_initial_inference_output_summary": {},
                    "lightzero_initial_inference_output_key_sample": [],
                    "lightzero_total_root_count": float(total_root_count),
                    "lightzero_filtered_zero_mask_root_count": float(dropped_zero_mask_root_count),
                    "lightzero_root_count": 0.0,
                    "lightzero_roots_per_call": 0.0,
                    "lightzero_policy_build_sec": float(
                        self._policy_metadata.get("build_sec", 0.0)
                    ),
                    "lightzero_policy_requested_cuda": bool(
                        self._policy_metadata.get("requested_cuda", False)
                    ),
                    "lightzero_policy_cuda": bool(self._policy_metadata.get("cuda", False)),
                }
            )

        started = time.perf_counter()
        obs_tensor = torch.as_tensor(flat_stack, dtype=torch.float32, device=device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        h2d_sec = time.perf_counter() - started

        started = time.perf_counter()
        if stack.dtype == np.uint8:
            obs_tensor = obs_tensor * (1.0 / 255.0)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
        normalize_sec = time.perf_counter() - started

        started = time.perf_counter()
        with torch.no_grad():
            output = model.initial_inference(obs_tensor)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        inference_sec = time.perf_counter() - started

        started = time.perf_counter()
        output_summary = _network_output_shape_summary(output)
        plain_summary = _plain_lightzero_value(output_summary)
        try:
            output_bytes = len(pickle.dumps(plain_summary, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            output_bytes = 0
        readback_sec = time.perf_counter() - started
        total_sec = tensor_prepare_sec + h2d_sec + normalize_sec + inference_sec + readback_sec

        return HybridBatchedStackProbeResult(
            telemetry={
                "host_to_device_sec": h2d_sec,
                "host_to_device_bytes": float(getattr(flat_stack, "nbytes", 0)),
                "normalize_sec": normalize_sec,
                "device_sec": inference_sec,
                "readback_sec": readback_sec,
                "total_sec": total_sec,
                "simulations": 0.0,
                "channels": 0.0,
                "roots": float(active_root_count),
                "input_rank": float(len(stack_shape)),
                "input_bytes": float(getattr(stack, "nbytes", 0)),
                "model_eval_count": float(active_root_count),
                "output_readback_bytes": float(output_bytes),
                "action_mask_consumed": 1.0,
                "compile_excluded_by_warmup": 1.0,
                "lightzero_initial_inference_total_sec": total_sec,
                "lightzero_initial_inference_tensor_prepare_sec": tensor_prepare_sec,
                "lightzero_initial_inference_h2d_sec": h2d_sec,
                "lightzero_initial_inference_normalize_sec": normalize_sec,
                "lightzero_initial_inference_forward_sec": inference_sec,
                "lightzero_initial_inference_readback_sec": readback_sec,
                "lightzero_initial_inference_stack_h2d_bytes": float(
                    getattr(flat_stack, "nbytes", 0)
                ),
                "lightzero_initial_inference_policy_device": str(device),
                "lightzero_initial_inference_policy_class": str(
                    self._policy_metadata.get("policy_class", type(self._policy).__name__)
                ),
                "lightzero_initial_inference_model_class": str(
                    self._policy_metadata.get("model_class", type(model).__name__)
                ),
                "lightzero_initial_inference_policy_surface": dict(
                    self._policy_metadata.get("surface", {})
                ),
                "lightzero_initial_inference_output_summary": plain_summary,
                "lightzero_initial_inference_output_key_sample": _output_key_sample_from_plain(
                    plain_summary
                ),
                "lightzero_total_root_count": float(total_root_count),
                "lightzero_filtered_zero_mask_root_count": float(dropped_zero_mask_root_count),
                "lightzero_root_count": float(active_root_count),
                "lightzero_roots_per_call": float(active_root_count),
                "lightzero_policy_build_sec": float(self._policy_metadata.get("build_sec", 0.0)),
                "lightzero_policy_requested_cuda": bool(
                    self._policy_metadata.get("requested_cuda", False)
                ),
                "lightzero_policy_cuda": bool(self._policy_metadata.get("cuda", False)),
            }
        )


class _LightZeroArrayCeilingStackProbe:
    """Profile-only ceiling probe for compact arrays around the LightZero MCTS branch."""

    backend_name = "lightzero_array_ceiling_consumer"
    semantics = "lightzero_replacement_ceiling_not_mcts"

    def __init__(
        self,
        *,
        policy: Any,
        policy_metadata: Mapping[str, Any],
        num_simulations: int,
        mode: str,
        input_mode: str = LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8,
        materialize_public_output: bool = False,
        require_resident_observation: bool = False,
        compile_search: bool = True,
        compile_model_inference: bool = False,
        require_model_compile: bool = False,
        model_compile_mode: str = "reduce-overhead",
        recurrent_action_shape_mode: str = "auto",
        timing_mode: str = "host_phase_sync",
        initial_inference_mode: str = "model_method",
        observation_memory_format: str = "contiguous",
        model_memory_format: str = "contiguous",
    ) -> None:
        if mode not in LIGHTZERO_ARRAY_CEILING_MODES:
            allowed = ", ".join(LIGHTZERO_ARRAY_CEILING_MODES)
            raise ValueError(f"mode must be one of {allowed}; got {mode!r}")
        if input_mode not in LIGHTZERO_ARRAY_CEILING_INPUT_MODES:
            allowed = ", ".join(LIGHTZERO_ARRAY_CEILING_INPUT_MODES)
            raise ValueError(f"input_mode must be one of {allowed}; got {input_mode!r}")
        self._require_resident_observation = bool(require_resident_observation)
        self._compact_torch_compile_search = bool(compile_search)
        self._compact_torch_compile_model_inference = bool(compile_model_inference)
        self._compact_torch_require_model_compile = bool(require_model_compile)
        self._compact_torch_model_compile_mode = str(model_compile_mode)
        self._compact_torch_recurrent_action_shape_mode = str(recurrent_action_shape_mode)
        self._compact_torch_timing_mode = str(timing_mode)
        self._compact_torch_initial_inference_mode = str(initial_inference_mode)
        self._compact_torch_observation_memory_format = str(observation_memory_format)
        self._compact_torch_model_memory_format = str(model_memory_format)
        if self._compact_torch_model_compile_mode not in COMPACT_TORCH_MODEL_COMPILE_MODES:
            allowed = ", ".join(COMPACT_TORCH_MODEL_COMPILE_MODES)
            raise ValueError(
                "model_compile_mode must be one of "
                f"{allowed}; got {self._compact_torch_model_compile_mode!r}"
            )
        if self._compact_torch_recurrent_action_shape_mode not in {"auto", "flat", "column"}:
            raise ValueError(
                "recurrent_action_shape_mode must be auto, flat, or column; "
                f"got {self._compact_torch_recurrent_action_shape_mode!r}"
            )
        if self._compact_torch_timing_mode not in COMPACT_TORCH_TIMING_MODES:
            allowed = ", ".join(COMPACT_TORCH_TIMING_MODES)
            raise ValueError(
                f"timing_mode must be one of {allowed}; got {self._compact_torch_timing_mode!r}"
            )
        if self._compact_torch_initial_inference_mode not in COMPACT_TORCH_INITIAL_INFERENCE_MODES:
            allowed = ", ".join(COMPACT_TORCH_INITIAL_INFERENCE_MODES)
            raise ValueError(
                "initial_inference_mode must be one of "
                f"{allowed}; got {self._compact_torch_initial_inference_mode!r}"
            )
        if self._compact_torch_observation_memory_format not in COMPACT_TORCH_MEMORY_FORMATS:
            allowed = ", ".join(COMPACT_TORCH_MEMORY_FORMATS)
            raise ValueError(
                "observation_memory_format must be one of "
                f"{allowed}; got {self._compact_torch_observation_memory_format!r}"
            )
        if self._compact_torch_model_memory_format not in COMPACT_TORCH_MEMORY_FORMATS:
            allowed = ", ".join(COMPACT_TORCH_MEMORY_FORMATS)
            raise ValueError(
                "model_memory_format must be one of "
                f"{allowed}; got {self._compact_torch_model_memory_format!r}"
            )
        if self._compact_torch_model_memory_format != "contiguous":
            raise ValueError(
                "model_memory_format=channels_last is parked for the current "
                "LightZero MuZero model because recurrent dynamics uses .view(); "
                "use model_memory_format='contiguous'"
            )
        if (
            self._compact_torch_compile_model_inference or self._compact_torch_require_model_compile
        ) and mode != LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE:
            raise ValueError(
                "compact Torch model compile flags are only supported by "
                "compact_torch_search_service"
            )
        if self._require_resident_observation and (
            mode != LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
        ):
            raise ValueError(
                "require_resident_observation is only supported by compact_torch_search_service"
            )
        if (
            mode == LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
            and input_mode != LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8
            and not self._require_resident_observation
        ):
            raise ValueError(
                "compact_torch_search_service consumes CompactRootBatchV1 root inputs; "
                "set hybrid_resident_observation_search for the fresh resident-device path"
            )
        if (
            mode == LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
            and input_mode != LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8
        ):
            raise ValueError(
                "fixed_shape_search_owner currently consumes CompactRootBatchV1 "
                "host uint8 observations directly; other input modes would be mislabeled"
            )
        self._policy = policy
        self._policy_metadata = dict(policy_metadata)
        self._num_simulations = _positive_int(
            num_simulations,
            "hybrid_lightzero_consumer_num_simulations",
        )
        self._mode = mode
        self._input_mode = input_mode
        self._materialize_public_output = bool(materialize_public_output)
        if (
            self._materialize_public_output
            and self._mode != LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE
        ):
            raise ValueError("materialize_public_output is only supported for mock_search_service")
        if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE:
            self.semantics = "mock_search_service_compact_arrays_profile_not_mcts"
        if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE:
            self.semantics = "service_tax_probe_compact_arrays_profile_not_mcts"
        if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS:
            self.semantics = "dense_torch_mcts_profile_not_lightzero_ctree"
        if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE:
            self.semantics = "dense_torch_mcts_compile_spike_profile_not_lightzero_ctree"
        if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE:
            self.semantics = "compact_torch_search_service_profile_not_lightzero_ctree"
        if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER:
            self.semantics = "fixed_shape_search_owner_first_legal_profile_not_mcts"
        self._resident_obs_tensor: Any | None = None
        self._resident_obs_signature: tuple[Any, ...] | None = None
        self._dense_compile_select_helper: Any | None = None
        self._dense_compile_backup_helper: Any | None = None
        self._dense_compile_signature: tuple[Any, ...] | None = None
        self._compact_torch_search_service_instance: CompactTorchSearchServiceV1 | None = None
        self._fixed_shape_search_owner: FixedShapeBatchedSearchOwnerV0 | None = None
        self._last_compact_search_arrays: dict[str, Any] | None = None
        self._last_compact_service_root_batch: Any | None = None
        self._last_compact_service_search_result: Any | None = None

    def run(self, observation: Any, action_mask: Any) -> HybridBatchedStackProbeResult:
        import numpy as np
        import torch

        self._last_compact_search_arrays = None
        model = getattr(self._policy, "_model", None)
        if model is None or not hasattr(model, "initial_inference"):
            raise ValueError("LightZero array-ceiling probe requires policy._model")

        stack = np.asarray(observation)
        stack_shape = tuple(int(dim) for dim in stack.shape)
        if stack_shape[2:] != (POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE):
            raise ValueError(
                "LightZero array-ceiling probe expects observation shape "
                f"[B,P,{POLICY_FRAME_STACK_DEPTH},{TARGET_SIZE},{TARGET_SIZE}], "
                f"got {stack_shape}"
            )
        mask = np.asarray(action_mask, dtype=np.float32)
        expected_mask_shape = stack_shape[:2] + (ACTION_COUNT,)
        if tuple(int(dim) for dim in mask.shape) != expected_mask_shape:
            raise ValueError(
                "LightZero array-ceiling probe action_mask shape mismatch; "
                f"got {mask.shape}, expected {expected_mask_shape}"
            )

        total_root_count = int(stack_shape[0] * stack_shape[1])
        flat_mask_all = mask.reshape((total_root_count, ACTION_COUNT))
        if not np.all((flat_mask_all == 0.0) | (flat_mask_all == 1.0)):
            raise ValueError("LightZero array-ceiling probe requires binary action masks")
        legal_root_mask = np.any(flat_mask_all > 0.0, axis=1)
        active_root_count = int(np.count_nonzero(legal_root_mask))
        dropped_zero_mask_root_count = int(total_root_count - active_root_count)
        device = _policy_model_device(self._policy)

        started = time.perf_counter()
        flat_stack_all = stack.reshape(
            (total_root_count, POLICY_FRAME_STACK_DEPTH, TARGET_SIZE, TARGET_SIZE)
        )
        if active_root_count == total_root_count:
            flat_stack = flat_stack_all
            flat_mask = flat_mask_all
            all_roots_legal_fast_path = True
        else:
            flat_stack = flat_stack_all[legal_root_mask]
            flat_mask = flat_mask_all[legal_root_mask]
            all_roots_legal_fast_path = False
        all_actions_legal_fast_path = bool(np.all(flat_mask > 0.0))
        tensor_prepare_sec = time.perf_counter() - started

        if active_root_count == 0:
            return HybridBatchedStackProbeResult(
                telemetry=self._zero_root_telemetry(
                    stack=stack,
                    mask=mask,
                    tensor_prepare_sec=tensor_prepare_sec,
                    total_root_count=total_root_count,
                    dropped_zero_mask_root_count=dropped_zero_mask_root_count,
                    device=device,
                )
            )

        (
            obs_tensor,
            h2d_sec,
            normalize_sec,
            input_telemetry,
        ) = self._prepare_observation_tensor(
            np=np,
            torch=torch,
            flat_stack=flat_stack,
            stack_dtype=stack.dtype,
            device=device,
        )

        started = time.perf_counter()
        with torch.no_grad():
            output = model.initial_inference(obs_tensor)
        _sync_torch_device_if_cuda(torch=torch, device=device)
        initial_inference_sec = time.perf_counter() - started

        started = time.perf_counter()
        policy_logits = _network_output_field(output, ("policy_logits", "policy"))
        if policy_logits is None:
            raise ValueError("array-ceiling probe requires policy_logits from initial_inference")
        root_value = _network_output_field(output, ("value", "predicted_value"))
        actions = root_values = priors = None
        if self._mode not in LIGHTZERO_ARRAY_CEILING_DENSE_TORCH_MCTS_MODES:
            actions, root_values, priors = _masked_policy_arrays(
                np=np,
                torch=torch,
                policy_logits=policy_logits,
                value=root_value,
                flat_mask=flat_mask,
                device=device,
            )
        visits = None
        recurrent_inference_sec = 0.0
        search_update_sec = 0.0
        recurrent_calls = 0
        latent_state = _network_output_field(output, ("latent_state", "hidden_state"))
        action_shape_mode = "none"
        compile_telemetry = _dense_torch_mcts_compile_default_telemetry(self._mode)
        output_assembly_sec = time.perf_counter() - started

        if self._mode in LIGHTZERO_ARRAY_CEILING_DENSE_TORCH_MCTS_MODES:
            dense_output = self._run_dense_torch_mcts(
                torch=torch,
                model=model,
                root_output=output,
                root_policy_logits=policy_logits,
                root_value=root_value,
                root_latent_state=latent_state,
                flat_mask=flat_mask,
                active_root_count=active_root_count,
                all_roots_legal_fast_path=all_roots_legal_fast_path,
                all_actions_legal_fast_path=all_actions_legal_fast_path,
                device=device,
            )
            actions = dense_output["actions"]
            root_values = dense_output["root_values"]
            priors = dense_output["visit_distributions"]
            recurrent_inference_sec = float(dense_output["recurrent_inference_sec"])
            search_update_sec = float(dense_output["search_update_sec"])
            recurrent_calls = int(dense_output["recurrent_calls"])
            action_shape_mode = str(dense_output["action_shape_mode"])
            compile_telemetry = dict(dense_output["compile_telemetry"])
            output_assembly_sec += float(dense_output["output_assembly_sec"])

        if self._mode in {
            LIGHTZERO_ARRAY_CEILING_MODE_RECURRENT_TOY,
            LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE,
        }:
            if latent_state is None:
                raise ValueError(f"{self._mode} requires latent_state from initial_inference")
            visits = _zeros_like_policy_array(
                np=np,
                torch=torch,
                root_count=active_root_count,
                device=device,
            )
            value_accum = _zeros_like_value_array(
                np=np,
                torch=torch,
                root_count=active_root_count,
                device=device,
            )
            for _sim in range(self._num_simulations):
                action_input = _recurrent_action_input(
                    np=np,
                    torch=torch,
                    actions=actions,
                    device=device,
                    mode=action_shape_mode,
                )
                started = time.perf_counter()
                with torch.no_grad():
                    try:
                        output = model.recurrent_inference(latent_state, action_input)
                        if action_shape_mode == "none":
                            action_shape_mode = "flat"
                    except Exception:
                        action_input = _recurrent_action_input(
                            np=np,
                            torch=torch,
                            actions=actions,
                            device=device,
                            mode="column",
                        )
                        output = model.recurrent_inference(latent_state, action_input)
                        action_shape_mode = "column"
                _sync_torch_device_if_cuda(torch=torch, device=device)
                recurrent_inference_sec += time.perf_counter() - started
                recurrent_calls += 1

                started = time.perf_counter()
                policy_logits = _network_output_field(output, ("policy_logits", "policy"))
                if policy_logits is None:
                    raise ValueError(
                        "recurrent_toy requires policy_logits from recurrent_inference"
                    )
                value = _network_output_field(output, ("value", "predicted_value"))
                actions, step_values, _priors = _masked_policy_arrays(
                    np=np,
                    torch=torch,
                    policy_logits=policy_logits,
                    value=value,
                    flat_mask=flat_mask,
                    device=device,
                )
                _scatter_visit_and_value(
                    np=np,
                    torch=torch,
                    visits=visits,
                    value_accum=value_accum,
                    actions=actions,
                    values=step_values,
                )
                next_latent_state = _network_output_field(
                    output,
                    ("latent_state", "hidden_state"),
                )
                if next_latent_state is not None:
                    latent_state = next_latent_state
                search_update_sec += time.perf_counter() - started
            if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE:
                if _is_torch_like_tensor(visits):
                    visit_totals = visits.sum(dim=1, keepdim=True).clamp_min(1.0)
                    priors = visits / visit_totals
                    root_values = value_accum / visit_totals.reshape(-1)
                else:
                    visit_totals = visits.sum(axis=1, keepdims=True)
                    priors = visits / np.maximum(visit_totals, np.float32(1.0))
                    root_values = value_accum / np.maximum(
                        visit_totals.reshape(-1),
                        np.float32(1.0),
                    )
            else:
                root_values = value_accum
                priors = visits

        started = time.perf_counter()
        actions_np = _array_to_numpy(np=np, value=actions).astype(np.int64, copy=False)
        values_np = _array_to_numpy(np=np, value=root_values).astype(np.float32, copy=False)
        priors_np = _array_to_numpy(np=np, value=priors).astype(np.float32, copy=False)
        _sync_torch_device_if_cuda(torch=torch, device=device)
        readback_sec = time.perf_counter() - started

        illegal_action_count = int(
            sum(
                1
                for row, action in enumerate(actions_np.tolist())
                if action < 0 or action >= ACTION_COUNT or float(flat_mask[row, action]) <= 0.0
            )
        )
        if illegal_action_count:
            raise ValueError(
                "LightZero array-ceiling probe decoded illegal actions: "
                f"{illegal_action_count} / {active_root_count}"
            )
        action_checksum = int(
            sum((index + 1) * (int(action) + 1) for index, action in enumerate(actions_np))
        )
        value_checksum = float(values_np.sum()) if values_np.size else 0.0
        output_bytes = int(actions_np.nbytes + values_np.nbytes + priors_np.nbytes)
        if self._mode in {
            LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE,
            LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE,
            *LIGHTZERO_ARRAY_CEILING_DENSE_TORCH_MCTS_MODES,
        }:
            actual_search_simulations = (
                0
                if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE
                else int(self._num_simulations)
            )
            self._last_compact_search_arrays = {
                "selected_action": actions_np.astype(np.int16, copy=True),
                "visit_policy": priors_np.astype(np.float32, copy=True),
                "root_value": values_np.astype(np.float32, copy=True),
                "array_source": "array_ceiling_compact_search",
                "search_impl": str(self._mode),
                "actual_search_simulations": actual_search_simulations,
                "requested_simulations": int(self._num_simulations),
            }
        public_output_sec = 0.0
        public_output_count = 0
        public_output_bytes = 0
        public_output_checksum = 0.0
        if self._materialize_public_output:
            started = time.perf_counter()
            policy_logits_np = (
                _array_to_numpy(np=np, value=policy_logits)
                .astype(np.float32, copy=False)
                .reshape(active_root_count, -1)[:, :ACTION_COUNT]
            )
            visit_counts_np = priors_np.astype(np.float32, copy=False) * np.float32(
                max(1, int(self._num_simulations))
            )
            safe_priors = np.where(priors_np > 0.0, priors_np, 1.0)
            entropies_np = (
                -np.sum(
                    np.where(
                        priors_np > 0.0,
                        priors_np * np.log2(safe_priors),
                        0.0,
                    ),
                    axis=1,
                )
                .astype(np.float32, copy=False)
                .reshape(active_root_count)
            )
            public_output = []
            for row in range(active_root_count):
                action = int(actions_np[row])
                public_output.append(
                    {
                        "action": action,
                        "visit_count_distributions": visit_counts_np[row].tolist(),
                        "visit_count_distribution_entropy": float(entropies_np[row]),
                        "searched_value": float(values_np[row]),
                        "predicted_value": float(values_np[row]),
                        "predicted_policy_logits": policy_logits_np[row].tolist(),
                    }
                )
            public_output_count = len(public_output)
            public_output_bytes = int(
                actions_np.nbytes
                + values_np.nbytes
                + visit_counts_np.nbytes
                + entropies_np.nbytes
                + policy_logits_np.nbytes
            )
            public_output_checksum = float(
                sum(
                    (row + 1)
                    * (
                        int(item["action"])
                        + 1
                        + float(item["searched_value"])
                        + float(item["visit_count_distribution_entropy"])
                    )
                    for row, item in enumerate(public_output)
                )
            )
            public_output_sec = time.perf_counter() - started
        input_preprocess_sec = float(input_telemetry.get("host_prenormalize_sec", 0.0))
        total_sec = (
            tensor_prepare_sec
            + input_preprocess_sec
            + h2d_sec
            + normalize_sec
            + initial_inference_sec
            + recurrent_inference_sec
            + search_update_sec
            + output_assembly_sec
            + readback_sec
            + public_output_sec
        )
        mock_search_service_telemetry: dict[str, Any] = {}
        if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE:
            mock_search_service_telemetry = {
                "mock_search_service_total_sec": total_sec,
                "mock_search_service_tensor_prepare_sec": tensor_prepare_sec,
                "mock_search_service_h2d_sec": h2d_sec,
                "mock_search_service_normalize_sec": normalize_sec,
                "mock_search_service_initial_inference_sec": initial_inference_sec,
                "mock_search_service_mask_softmax_sec": output_assembly_sec,
                "mock_search_service_compact_output_sec": output_assembly_sec,
                "mock_search_service_readback_sec": readback_sec,
                "mock_search_service_public_output_sec": public_output_sec,
                "mock_search_service_input_bytes": float(getattr(flat_stack, "nbytes", 0))
                + float(getattr(flat_mask, "nbytes", 0)),
                "mock_search_service_compact_output_bytes": float(output_bytes),
                "mock_search_service_public_output_bytes": float(public_output_bytes),
                "mock_search_service_public_output_count": float(public_output_count),
                "mock_search_service_public_output_checksum": public_output_checksum,
                "mock_search_service_active_roots": float(active_root_count),
                "mock_search_service_zero_mask_roots": float(dropped_zero_mask_root_count),
                "mock_search_service_requested_simulations": float(self._num_simulations),
                "mock_search_service_actual_search_simulations": 0.0,
                "mock_search_service_recurrent_inference_calls": 0.0,
                "mock_search_service_real_ctree_calls": 0.0,
                "mock_search_service_illegal_action_count": float(illegal_action_count),
                "mock_search_service_action_checksum": float(action_checksum),
                "mock_search_service_value_checksum": value_checksum,
                "mock_search_service_visit_shape": list(priors_np.shape),
                "mock_search_service_semantics": self.semantics,
            }
        service_tax_telemetry: dict[str, Any] = {}
        if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE:
            pack_sec = tensor_prepare_sec + input_preprocess_sec + h2d_sec + normalize_sec
            service_call_sec = initial_inference_sec + recurrent_inference_sec + search_update_sec
            service_tax_telemetry = {
                "service_tax_probe_total_sec": total_sec,
                "service_tax_probe_pack_sec": pack_sec,
                "service_tax_probe_service_call_sec": service_call_sec,
                "service_tax_probe_initial_inference_sec": initial_inference_sec,
                "service_tax_probe_recurrent_inference_sec": recurrent_inference_sec,
                "service_tax_probe_recurrent_inference_calls": float(recurrent_calls),
                "service_tax_probe_fake_search_update_sec": search_update_sec,
                "service_tax_probe_unpack_sec": readback_sec,
                "service_tax_probe_compact_output_bytes": float(output_bytes),
                "service_tax_probe_input_bytes": float(getattr(flat_stack, "nbytes", 0))
                + float(getattr(flat_mask, "nbytes", 0)),
                "service_tax_probe_active_roots": float(active_root_count),
                "service_tax_probe_zero_mask_roots": float(dropped_zero_mask_root_count),
                "service_tax_probe_requested_simulations": float(self._num_simulations),
                "service_tax_probe_actual_search_simulations": float(self._num_simulations),
                "service_tax_probe_real_ctree_calls": 0.0,
                "service_tax_probe_illegal_action_count": float(illegal_action_count),
                "service_tax_probe_action_checksum": float(action_checksum),
                "service_tax_probe_value_checksum": value_checksum,
                "service_tax_probe_visit_shape": list(priors_np.shape),
                "service_tax_probe_compact_search_arrays_stored": float(
                    self._last_compact_search_arrays is not None
                ),
                "service_tax_probe_semantics": self.semantics,
            }

        return HybridBatchedStackProbeResult(
            telemetry={
                "host_to_device_sec": h2d_sec,
                "host_to_device_bytes": float(getattr(flat_stack, "nbytes", 0)),
                "normalize_sec": normalize_sec,
                "device_sec": initial_inference_sec + recurrent_inference_sec,
                "readback_sec": readback_sec,
                "total_sec": total_sec,
                "simulations": float(
                    self._num_simulations
                    if self._mode
                    in {
                        LIGHTZERO_ARRAY_CEILING_MODE_RECURRENT_TOY,
                        LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE,
                        *LIGHTZERO_ARRAY_CEILING_DENSE_TORCH_MCTS_MODES,
                    }
                    else 0
                ),
                "channels": 0.0,
                "roots": float(active_root_count),
                "input_rank": float(len(stack_shape)),
                "input_bytes": float(getattr(stack, "nbytes", 0)),
                "model_eval_count": float(active_root_count * (1 + recurrent_calls)),
                "output_readback_bytes": float(output_bytes),
                "action_mask_consumed": 1.0,
                "compile_excluded_by_warmup": 1.0,
                "lightzero_array_ceiling_mode": self._mode,
                "lightzero_array_ceiling_semantics": self.semantics,
                "lightzero_array_ceiling_input_mode": self._input_mode,
                "lightzero_array_ceiling_all_roots_legal_fast_path": float(
                    all_roots_legal_fast_path
                ),
                "lightzero_array_ceiling_all_actions_legal_fast_path": float(
                    all_actions_legal_fast_path
                ),
                "lightzero_array_ceiling_recurrent_sync_mode": (
                    "final_readback_only"
                    if self._mode in LIGHTZERO_ARRAY_CEILING_DENSE_TORCH_MCTS_MODES
                    else "per_recurrent_call"
                ),
                **compile_telemetry,
                **mock_search_service_telemetry,
                **service_tax_telemetry,
                "lightzero_array_ceiling_total_sec": total_sec,
                "lightzero_array_ceiling_tensor_prepare_sec": tensor_prepare_sec,
                "lightzero_array_ceiling_h2d_sec": h2d_sec,
                "lightzero_array_ceiling_normalize_sec": normalize_sec,
                "lightzero_array_ceiling_pin_memory_sec": float(
                    input_telemetry.get("pin_memory_sec", 0.0)
                ),
                "lightzero_array_ceiling_host_prenormalize_sec": float(
                    input_telemetry.get("host_prenormalize_sec", 0.0)
                ),
                "lightzero_array_ceiling_resident_first_fill_sec": float(
                    input_telemetry.get("resident_first_fill_sec", 0.0)
                ),
                "lightzero_array_ceiling_resident_reused": float(
                    input_telemetry.get("resident_reused", 0.0)
                ),
                "lightzero_array_ceiling_initial_inference_sec": initial_inference_sec,
                "lightzero_array_ceiling_initial_inference_calls": 1.0,
                "lightzero_array_ceiling_recurrent_inference_sec": recurrent_inference_sec,
                "lightzero_array_ceiling_recurrent_inference_calls": float(recurrent_calls),
                "lightzero_array_ceiling_requested_simulations": float(self._num_simulations),
                "lightzero_array_ceiling_actual_search_simulations": float(
                    self._last_compact_search_arrays.get("actual_search_simulations", 0)
                    if self._last_compact_search_arrays is not None
                    else self._num_simulations
                    if self._mode in LIGHTZERO_ARRAY_CEILING_DENSE_TORCH_MCTS_MODES
                    else 0
                ),
                "lightzero_array_ceiling_compact_search_arrays_stored": float(
                    self._last_compact_search_arrays is not None
                ),
                "lightzero_array_ceiling_search_update_sec": search_update_sec,
                "lightzero_array_ceiling_output_assembly_sec": output_assembly_sec,
                "lightzero_array_ceiling_readback_sec": readback_sec,
                "lightzero_array_ceiling_public_output_materialized": float(
                    self._materialize_public_output
                ),
                "lightzero_array_ceiling_public_output_sec": public_output_sec,
                "lightzero_array_ceiling_public_output_count": float(public_output_count),
                "lightzero_array_ceiling_public_output_bytes": float(public_output_bytes),
                "lightzero_array_ceiling_output_bytes": float(output_bytes),
                "lightzero_array_ceiling_policy_shape": list(priors_np.shape),
                "lightzero_array_ceiling_value_shape": list(values_np.shape),
                "lightzero_array_ceiling_action_shape": list(actions_np.shape),
                "lightzero_array_ceiling_action_checksum": float(action_checksum),
                "lightzero_array_ceiling_value_checksum": value_checksum,
                "lightzero_array_ceiling_illegal_action_count": float(illegal_action_count),
                "lightzero_array_ceiling_action_shape_mode": action_shape_mode,
                "lightzero_array_ceiling_policy_device": str(device),
                "lightzero_array_ceiling_policy_class": str(
                    self._policy_metadata.get("policy_class", type(self._policy).__name__)
                ),
                "lightzero_array_ceiling_model_class": str(
                    self._policy_metadata.get("model_class", type(model).__name__)
                ),
                "lightzero_array_ceiling_policy_surface": dict(
                    self._policy_metadata.get("surface", {})
                ),
                "lightzero_array_ceiling_first_actions": actions_np[: min(16, len(actions_np))]
                .astype(int)
                .tolist(),
                "lightzero_total_root_count": float(total_root_count),
                "lightzero_filtered_zero_mask_root_count": float(dropped_zero_mask_root_count),
                "lightzero_root_count": float(active_root_count),
                "lightzero_roots_per_call": float(active_root_count),
                "lightzero_policy_build_sec": float(self._policy_metadata.get("build_sec", 0.0)),
                "lightzero_policy_requested_cuda": bool(
                    self._policy_metadata.get("requested_cuda", False)
                ),
                "lightzero_policy_cuda": bool(self._policy_metadata.get("cuda", False)),
            }
        )

    def run_compact_batch(self, batch: HybridCompactBatch) -> HybridBatchedStackProbeResult:
        import numpy as np

        self._last_compact_service_root_batch = None
        self._last_compact_service_search_result = None
        if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE:
            return self._run_compact_torch_search_service_batch(batch, np=np)
        if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER:
            return self._run_fixed_shape_search_owner_batch(batch, np=np)
        result = self.run(batch.observation, batch.action_mask)
        telemetry = dict(result.telemetry)
        if self._last_compact_search_arrays is None:
            return HybridBatchedStackProbeResult(telemetry=telemetry)

        started = time.perf_counter()
        root_batch = build_compact_root_batch_v1(
            batch,
            search_lane=self._mode,
            metadata={
                "profile_backend": self.backend_name,
                "profile_semantics": self.semantics,
            },
            copy_observation=False,
            observation_source=str(
                getattr(batch, "observation_source", COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1)
            ),
            resident_observation=getattr(batch, "resident_observation", None),
        )
        search_result = compact_search_result_v1_from_arrays(
            root_batch,
            self._last_compact_search_arrays,
            default_search_impl=self._mode,
            default_num_simulations=self._num_simulations,
            metadata={
                "profile_backend": self.backend_name,
                "profile_semantics": self.semantics,
            },
        )
        self._last_compact_service_root_batch = root_batch
        self._last_compact_service_search_result = search_result
        telemetry.update(
            {
                "compact_service_contract_v1_enabled": True,
                "compact_service_contract_v1_validation_sec": time.perf_counter() - started,
                "compact_service_contract_v1_contract_id": str(
                    root_batch.metadata.get("contract_id")
                ),
                "compact_service_root_batch_schema_id": str(root_batch.metadata.get("schema_id")),
                "compact_service_search_result_schema_id": str(
                    search_result.metadata.get("schema_id")
                ),
                "compact_service_root_count": float(root_batch.legal_mask.shape[0]),
                "compact_service_active_root_count": float(search_result.root_index.size),
                "compact_service_selected_action_checksum": float(
                    np.asarray(search_result.selected_action, dtype=np.int64).sum()
                ),
                "compact_service_visit_policy_checksum": float(
                    np.asarray(search_result.visit_policy, dtype=np.float64).sum()
                ),
                "compact_service_identity_checksum": float(
                    np.asarray(search_result.env_row, dtype=np.int64).sum()
                    + np.asarray(search_result.player, dtype=np.int64).sum()
                ),
            }
        )
        return HybridBatchedStackProbeResult(telemetry=telemetry)

    def _compact_torch_search_service(self) -> CompactTorchSearchServiceV1:
        if self._compact_torch_search_service_instance is None:
            self._compact_torch_search_service_instance = (
                self.new_compact_torch_search_service_variant(
                    request_model_compile=self._compact_torch_compile_model_inference,
                    require_model_compile=self._compact_torch_require_model_compile,
                    model_compile_mode=self._compact_torch_model_compile_mode,
                )
            )
        return self._compact_torch_search_service_instance

    def new_compact_torch_search_service_variant(
        self,
        *,
        request_model_compile: bool,
        require_model_compile: bool,
        model_compile_mode: str,
        initial_inference_mode: str | None = None,
    ) -> CompactTorchSearchServiceV1:
        selected_initial_inference_mode = (
            self._compact_torch_initial_inference_mode
            if initial_inference_mode is None
            else str(initial_inference_mode)
        )
        return CompactTorchSearchServiceV1(
            policy=self._policy,
            num_simulations=self._num_simulations,
            compile_config=CompactTorchCompileConfig(
                request_compile=self._compact_torch_compile_search,
                require_cuda_device=False,
                require_torch_compile=False,
                require_all_roots_active=False,
                require_all_actions_legal=False,
                request_model_compile=bool(request_model_compile),
                require_model_compile=bool(require_model_compile),
                model_compile_mode=str(model_compile_mode),
                recurrent_action_shape_mode=(self._compact_torch_recurrent_action_shape_mode),
                timing_mode=self._compact_torch_timing_mode,
                initial_inference_mode=selected_initial_inference_mode,
                observation_memory_format=self._compact_torch_observation_memory_format,
                model_memory_format=self._compact_torch_model_memory_format,
            ),
            require_resident_observation=self._require_resident_observation,
        )

    def _run_compact_torch_search_service_root_batch(self, root_batch: Any) -> Any:
        search_result = self._compact_torch_search_service().run(root_batch)
        self._last_compact_service_root_batch = root_batch
        self._last_compact_service_search_result = search_result
        self._last_compact_search_arrays = {
            "selected_action": search_result.selected_action,
            "visit_policy": search_result.visit_policy,
            "root_value": search_result.root_value,
            "raw_visit_counts": search_result.raw_visit_counts,
            "search_impl": self._mode,
            "service_impl": COMPACT_TORCH_SEARCH_SERVICE_IMPL,
            "actual_search_simulations": int(self._num_simulations),
            "array_source": "compact_torch_search_service",
        }
        return search_result

    def _run_compact_torch_search_service_action_step(self, root_batch: Any) -> Any:
        import numpy as np

        action_step = self._compact_torch_search_service().run_action_step(root_batch)
        self._last_compact_service_root_batch = root_batch
        self._last_compact_service_search_result = None
        self._last_compact_search_arrays = {
            "selected_action": action_step.selected_action,
            "visit_policy": np.zeros(
                (int(action_step.selected_action.size), ACTION_COUNT),
                dtype=np.float32,
            ),
            "root_value": np.zeros(
                (int(action_step.selected_action.size),),
                dtype=np.float32,
            ),
            "raw_visit_counts": None,
            "search_impl": self._mode,
            "service_impl": COMPACT_TORCH_SEARCH_SERVICE_IMPL,
            "actual_search_simulations": int(self._num_simulations),
            "requested_simulations": int(self._num_simulations),
            "array_source": "compact_torch_search_service_action_step",
        }
        return action_step

    def _flush_compact_torch_search_service_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> Any:
        return self._compact_torch_search_service().flush_replay_payload(replay_payload_handle)

    def _flush_compact_torch_search_service_device_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> Any:
        return self._compact_torch_search_service().flush_device_replay_payload(
            replay_payload_handle
        )

    def _run_compact_torch_search_service_batch(
        self,
        batch: HybridCompactBatch,
        *,
        np: Any,
    ) -> HybridBatchedStackProbeResult:
        started = time.perf_counter()
        root_batch = build_compact_root_batch_v1(
            batch,
            search_lane=self._mode,
            metadata={
                "profile_backend": self.backend_name,
                "profile_semantics": self.semantics,
            },
            observation_source=str(
                getattr(batch, "observation_source", COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1)
            ),
            resident_observation=getattr(batch, "resident_observation", None),
        )
        search_result = self._run_compact_torch_search_service_root_batch(root_batch)
        total_sec = time.perf_counter() - started
        compact_output_bytes = int(
            search_result.selected_action.nbytes
            + search_result.visit_policy.nbytes
            + search_result.root_value.nbytes
            + (
                0
                if search_result.raw_visit_counts is None
                else search_result.raw_visit_counts.nbytes
            )
        )
        service_metadata = dict(getattr(search_result, "metadata", {}) or {})
        service_tensor_prepare_sec = float(
            service_metadata.get("compact_torch_search_service_tensor_prepare_sec", 0.0)
        )
        service_initial_inference_sec = float(
            service_metadata.get("compact_torch_search_service_initial_inference_sec", 0.0)
        )
        service_initial_inference_enqueue_sec = float(
            service_metadata.get(
                "compact_torch_search_service_initial_inference_enqueue_sec",
                0.0,
            )
        )
        service_initial_inference_sync_sec = float(
            service_metadata.get(
                "compact_torch_search_service_initial_inference_sync_sec",
                0.0,
            )
        )
        service_initial_inference_cuda_event_sec = float(
            service_metadata.get(
                "compact_torch_search_service_initial_inference_cuda_event_sec",
                0.0,
            )
        )
        service_tree_search_sec = float(
            service_metadata.get("compact_torch_search_service_tree_search_sec", total_sec)
        )
        service_tree_recurrent_inference_enqueue_sec = float(
            service_metadata.get(
                "compact_torch_search_service_tree_recurrent_inference_enqueue_sec",
                0.0,
            )
        )
        service_tree_sync_sec = float(
            service_metadata.get("compact_torch_search_service_tree_sync_sec", 0.0)
        )
        service_tree_recurrent_inference_cuda_event_sec = float(
            service_metadata.get(
                "compact_torch_search_service_tree_recurrent_inference_cuda_event_sec",
                0.0,
            )
        )
        service_tree_cuda_event_sec = float(
            service_metadata.get("compact_torch_search_service_tree_cuda_event_sec", 0.0)
        )
        service_readback_sec = float(
            service_metadata.get("compact_torch_search_service_readback_sec", 0.0)
        )
        service_root_noise_weight = float(
            service_metadata.get("compact_torch_search_root_noise_weight", 0.0)
        )
        service_obs_h2d_bytes = float(
            service_metadata.get("compact_torch_search_service_obs_h2d_bytes", 0.0)
        )
        service_mask_h2d_bytes = float(
            service_metadata.get("compact_torch_search_service_mask_h2d_bytes", 0.0)
        )
        service_action_d2h_bytes = float(
            service_metadata.get("compact_torch_search_service_action_d2h_bytes", 0.0)
        )
        service_replay_payload_d2h_bytes = float(
            service_metadata.get(
                "compact_torch_search_service_replay_payload_d2h_bytes",
                0.0,
            )
        )
        return HybridBatchedStackProbeResult(
            telemetry={
                "host_to_device_sec": 0.0,
                "host_to_device_bytes": service_obs_h2d_bytes + service_mask_h2d_bytes,
                "normalize_sec": 0.0,
                "device_sec": total_sec,
                "readback_sec": 0.0,
                "total_sec": total_sec,
                "simulations": float(self._num_simulations),
                "channels": 0.0,
                "roots": float(search_result.root_index.size),
                "input_rank": float(len(getattr(root_batch.observation, "shape", ()))),
                "input_bytes": float(getattr(root_batch.observation, "nbytes", 0)),
                "model_eval_count": float(
                    search_result.root_index.size * (1 + self._num_simulations)
                ),
                "output_readback_bytes": float(compact_output_bytes),
                "action_mask_consumed": 1.0,
                "compile_excluded_by_warmup": 1.0,
                "lightzero_array_ceiling_mode": self._mode,
                "lightzero_array_ceiling_semantics": self.semantics,
                "lightzero_array_ceiling_input_mode": self._input_mode,
                "lightzero_array_ceiling_total_sec": total_sec,
                "lightzero_array_ceiling_tensor_prepare_sec": service_tensor_prepare_sec,
                "lightzero_array_ceiling_h2d_sec": 0.0,
                "lightzero_array_ceiling_normalize_sec": 0.0,
                "lightzero_array_ceiling_initial_inference_sec": service_initial_inference_sec,
                "lightzero_array_ceiling_recurrent_inference_sec": 0.0,
                "lightzero_array_ceiling_search_update_sec": service_tree_search_sec,
                "lightzero_array_ceiling_output_assembly_sec": 0.0,
                "lightzero_array_ceiling_readback_sec": service_readback_sec,
                "compact_torch_search_service_initial_inference_enqueue_sec": (
                    service_initial_inference_enqueue_sec
                ),
                "compact_torch_search_service_initial_inference_sync_sec": (
                    service_initial_inference_sync_sec
                ),
                "compact_torch_search_service_initial_inference_cuda_event_sec": (
                    service_initial_inference_cuda_event_sec
                ),
                "compact_torch_search_service_initial_inference_cuda_event_status": str(
                    service_metadata.get(
                        "compact_torch_search_service_initial_inference_cuda_event_status",
                        "",
                    )
                ),
                "compact_torch_search_service_tree_recurrent_inference_enqueue_sec": (
                    service_tree_recurrent_inference_enqueue_sec
                ),
                "compact_torch_search_service_tree_recurrent_inference_cuda_event_sec": (
                    service_tree_recurrent_inference_cuda_event_sec
                ),
                "compact_torch_search_service_tree_recurrent_inference_cuda_event_status": str(
                    service_metadata.get(
                        "compact_torch_search_service_tree_recurrent_inference_cuda_event_status",
                        "",
                    )
                ),
                "compact_torch_search_service_tree_sync_sec": service_tree_sync_sec,
                "compact_torch_search_service_tree_cuda_event_sec": (service_tree_cuda_event_sec),
                "compact_torch_search_service_tree_cuda_event_status": str(
                    service_metadata.get(
                        "compact_torch_search_service_tree_cuda_event_status",
                        "",
                    )
                ),
                "compact_torch_search_service_timing_mode": str(
                    service_metadata.get("compact_torch_search_service_timing_mode", "")
                ),
                "compact_torch_search_service_cuda_event_timing_enabled": float(
                    bool(
                        service_metadata.get(
                            "compact_torch_search_service_cuda_event_timing_enabled",
                            False,
                        )
                    )
                ),
                "compact_torch_search_service_initial_sync_enabled": float(
                    bool(
                        service_metadata.get(
                            "compact_torch_search_service_initial_sync_enabled",
                            True,
                        )
                    )
                ),
                "lightzero_array_ceiling_obs_h2d_bytes": service_obs_h2d_bytes,
                "lightzero_array_ceiling_mask_h2d_bytes": service_mask_h2d_bytes,
                "lightzero_array_ceiling_action_d2h_bytes": service_action_d2h_bytes,
                "lightzero_array_ceiling_replay_payload_d2h_bytes": (
                    service_replay_payload_d2h_bytes
                ),
                "lightzero_array_ceiling_root_observation_copy_bytes": float(
                    service_metadata.get(
                        "compact_torch_search_service_root_observation_copy_bytes",
                        0.0,
                    )
                ),
                "lightzero_array_ceiling_python_rows_materialized": float(
                    service_metadata.get(
                        "compact_torch_search_service_python_rows_materialized",
                        0.0,
                    )
                ),
                "lightzero_array_ceiling_rnd_materialized_rows": float(
                    service_metadata.get(
                        "compact_torch_search_service_rnd_materialized_rows",
                        0.0,
                    )
                ),
                "lightzero_array_ceiling_resident_obs_reused": float(
                    service_metadata.get(
                        "compact_torch_search_service_resident_obs_reused",
                        0.0,
                    )
                ),
                "compact_torch_search_service_tree_search_includes_recurrent": True,
                "lightzero_array_ceiling_root_noise_weight": service_root_noise_weight,
                "lightzero_array_ceiling_compile_status": str(
                    service_metadata.get("compact_torch_search_compile_status", "")
                ),
                "lightzero_array_ceiling_compile_reason": str(
                    service_metadata.get("compact_torch_search_compile_reason", "")
                ),
                "lightzero_array_ceiling_compile_enabled": float(
                    service_metadata.get("compact_torch_search_compile_enabled", 0.0)
                ),
                "lightzero_array_ceiling_compile_attempted": float(
                    service_metadata.get("compact_torch_search_compile_attempted", 0.0)
                ),
                "lightzero_array_ceiling_model_compile_requested": float(
                    bool(
                        service_metadata.get(
                            "compact_torch_search_model_compile_requested",
                            False,
                        )
                    )
                ),
                "lightzero_array_ceiling_model_compile_attempted": float(
                    service_metadata.get(
                        "compact_torch_search_model_compile_attempted",
                        0.0,
                    )
                ),
                "lightzero_array_ceiling_model_compile_used": float(
                    bool(
                        service_metadata.get(
                            "compact_torch_search_model_compile_used",
                            False,
                        )
                    )
                ),
                "lightzero_array_ceiling_model_compile_cache_hit": float(
                    bool(
                        service_metadata.get(
                            "compact_torch_search_model_compile_cache_hit",
                            False,
                        )
                    )
                ),
                "lightzero_array_ceiling_model_compile_runtime_status": str(
                    service_metadata.get(
                        "compact_torch_search_model_compile_runtime_status",
                        "",
                    )
                ),
                "lightzero_array_ceiling_recurrent_action_shape_mode_effective": str(
                    service_metadata.get(
                        "compact_torch_search_recurrent_action_shape_mode_effective",
                        "",
                    )
                ),
                "lightzero_array_ceiling_recurrent_action_shape_fallback_count": float(
                    service_metadata.get(
                        "compact_torch_search_recurrent_action_shape_exception_fallback_count",
                        0.0,
                    )
                ),
                "lightzero_array_ceiling_compact_output_bytes": float(compact_output_bytes),
                "lightzero_array_ceiling_active_roots": float(search_result.root_index.size),
                "lightzero_array_ceiling_zero_mask_roots": float(
                    root_batch.legal_mask.shape[0] - search_result.root_index.size
                ),
                "lightzero_array_ceiling_requested_simulations": float(self._num_simulations),
                "lightzero_array_ceiling_actual_search_simulations": float(self._num_simulations),
                "lightzero_array_ceiling_recurrent_inference_calls": float(self._num_simulations),
                "lightzero_array_ceiling_real_ctree_calls": 0.0,
                "lightzero_array_ceiling_illegal_action_count": 0.0,
                "lightzero_array_ceiling_action_checksum": float(
                    np.asarray(search_result.selected_action, dtype=np.int64).sum()
                ),
                "lightzero_array_ceiling_value_checksum": float(
                    np.asarray(search_result.root_value, dtype=np.float64).sum()
                ),
                "lightzero_array_ceiling_visit_shape": list(search_result.visit_policy.shape),
                "lightzero_array_ceiling_compact_search_arrays_stored": 1.0,
                "lightzero_array_ceiling_policy_device": str(
                    search_result.metadata.get("compact_torch_search_service_device", "")
                ),
                "lightzero_array_ceiling_policy_class": str(
                    self._policy_metadata.get("policy_class", type(self._policy).__name__)
                ),
                "lightzero_array_ceiling_model_class": str(
                    self._policy_metadata.get(
                        "model_class",
                        type(getattr(self._policy, "_model", self._policy)).__name__,
                    )
                ),
                "lightzero_array_ceiling_policy_surface": dict(
                    self._policy_metadata.get("surface", {})
                ),
                "lightzero_array_ceiling_first_actions": search_result.selected_action[
                    : min(16, len(search_result.selected_action))
                ]
                .astype(int)
                .tolist(),
                "compact_service_contract_v1_enabled": True,
                "compact_service_contract_v1_validation_sec": 0.0,
                "compact_service_contract_v1_contract_id": str(
                    root_batch.metadata.get("contract_id")
                ),
                "compact_service_root_batch_schema_id": str(root_batch.metadata.get("schema_id")),
                "compact_service_search_result_schema_id": str(
                    search_result.metadata.get("schema_id")
                ),
                "compact_service_root_count": float(root_batch.legal_mask.shape[0]),
                "compact_service_active_root_count": float(search_result.root_index.size),
                "compact_service_selected_action_checksum": float(
                    np.asarray(search_result.selected_action, dtype=np.int64).sum()
                ),
                "compact_service_visit_policy_checksum": float(
                    np.asarray(search_result.visit_policy, dtype=np.float64).sum()
                ),
                "compact_service_search_impl": COMPACT_TORCH_SEARCH_SERVICE_IMPL,
                "compact_service_profile_mode": self._mode,
            }
        )

    def _fixed_shape_search_owner_for_root_batch(
        self,
        root_batch: Any,
    ) -> FixedShapeBatchedSearchOwnerV0:
        root_count = int(root_batch.legal_mask.shape[0])
        if self._fixed_shape_search_owner is None:
            self._fixed_shape_search_owner = FixedShapeBatchedSearchOwnerV0(
                root_count=root_count,
                num_simulations=self._num_simulations,
            )
        elif self._fixed_shape_search_owner.root_count != root_count:
            raise ValueError(
                "fixed_shape_search_owner requires stable root_count; "
                f"got {root_count}, expected {self._fixed_shape_search_owner.root_count}"
            )
        return self._fixed_shape_search_owner

    def _run_fixed_shape_search_owner_root_batch(self, root_batch: Any) -> Any:
        owner = self._fixed_shape_search_owner_for_root_batch(root_batch)
        search_result = owner.run(root_batch)
        self._last_compact_service_root_batch = root_batch
        self._last_compact_service_search_result = search_result
        self._last_compact_search_arrays = {
            "selected_action": search_result.selected_action,
            "visit_policy": search_result.visit_policy,
            "root_value": search_result.root_value,
            "raw_visit_counts": search_result.raw_visit_counts,
            "search_impl": self._mode,
            "service_impl": FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL,
            "actual_search_simulations": 0,
            "requested_simulations": int(self._num_simulations),
            "array_source": "fixed_shape_search_owner",
        }
        return search_result

    def _run_fixed_shape_search_owner_action_step(self, root_batch: Any) -> Any:
        import numpy as np

        owner = self._fixed_shape_search_owner_for_root_batch(root_batch)
        action_step = owner.run_action_step(root_batch)
        self._last_compact_service_root_batch = root_batch
        self._last_compact_service_search_result = None
        self._last_compact_search_arrays = {
            "selected_action": action_step.selected_action,
            "visit_policy": np.zeros(
                (int(action_step.selected_action.size), ACTION_COUNT),
                dtype=np.float32,
            ),
            "root_value": np.zeros(
                (int(action_step.selected_action.size),),
                dtype=np.float32,
            ),
            "raw_visit_counts": None,
            "search_impl": self._mode,
            "service_impl": FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL,
            "actual_search_simulations": 0,
            "requested_simulations": int(self._num_simulations),
            "array_source": "fixed_shape_search_owner_action_step",
        }
        return action_step

    def _flush_fixed_shape_search_owner_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> Any:
        if self._fixed_shape_search_owner is None:
            raise ValueError("fixed_shape_search_owner replay payload requested before run")
        return self._fixed_shape_search_owner.flush_replay_payload(replay_payload_handle)

    def _run_fixed_shape_search_owner_batch(
        self,
        batch: HybridCompactBatch,
        *,
        np: Any,
    ) -> HybridBatchedStackProbeResult:
        started = time.perf_counter()
        root_batch = build_compact_root_batch_v1(
            batch,
            search_lane=self._mode,
            metadata={
                "profile_backend": self.backend_name,
                "profile_semantics": self.semantics,
            },
            observation_source=str(
                getattr(batch, "observation_source", COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1)
            ),
            resident_observation=getattr(batch, "resident_observation", None),
        )
        search_result = self._run_fixed_shape_search_owner_root_batch(root_batch)
        total_sec = time.perf_counter() - started
        owner_metadata = dict(getattr(search_result, "metadata", {}) or {})
        owner_profile = owner_metadata.get("profile_telemetry")
        owner_profile = owner_profile if isinstance(owner_profile, Mapping) else {}
        compact_output_bytes = int(
            search_result.selected_action.nbytes
            + search_result.visit_policy.nbytes
            + search_result.root_value.nbytes
            + (
                0
                if search_result.raw_visit_counts is None
                else search_result.raw_visit_counts.nbytes
            )
        )
        owner_total_sec = float(
            owner_profile.get(
                "fixed_shape_batched_search_owner_total_sec",
                total_sec,
            )
        )
        owner_d2h_bytes = float(
            owner_profile.get("fixed_shape_batched_search_owner_d2h_bytes", 0.0)
        )
        owner_preallocated_bytes = float(
            owner_profile.get(
                "fixed_shape_batched_search_owner_preallocated_buffer_bytes",
                0.0,
            )
        )
        return HybridBatchedStackProbeResult(
            telemetry={
                "host_to_device_sec": 0.0,
                "host_to_device_bytes": 0.0,
                "normalize_sec": 0.0,
                "device_sec": owner_total_sec,
                "readback_sec": 0.0,
                "total_sec": total_sec,
                "simulations": 0.0,
                "channels": 0.0,
                "roots": float(search_result.root_index.size),
                "input_rank": float(len(getattr(root_batch.observation, "shape", ()))),
                "input_bytes": float(getattr(root_batch.observation, "nbytes", 0)),
                "model_eval_count": 0.0,
                "output_readback_bytes": float(compact_output_bytes),
                "action_mask_consumed": 1.0,
                "compile_excluded_by_warmup": 1.0,
                "lightzero_array_ceiling_mode": self._mode,
                "lightzero_array_ceiling_semantics": self.semantics,
                "lightzero_array_ceiling_input_mode": self._input_mode,
                "lightzero_array_ceiling_total_sec": total_sec,
                "lightzero_array_ceiling_tensor_prepare_sec": 0.0,
                "lightzero_array_ceiling_h2d_sec": 0.0,
                "lightzero_array_ceiling_normalize_sec": 0.0,
                "lightzero_array_ceiling_initial_inference_sec": 0.0,
                "lightzero_array_ceiling_recurrent_inference_sec": 0.0,
                "lightzero_array_ceiling_search_update_sec": owner_total_sec,
                "lightzero_array_ceiling_output_assembly_sec": 0.0,
                "lightzero_array_ceiling_readback_sec": 0.0,
                "lightzero_array_ceiling_obs_h2d_bytes": 0.0,
                "lightzero_array_ceiling_mask_h2d_bytes": 0.0,
                "lightzero_array_ceiling_action_d2h_bytes": float(
                    search_result.selected_action.nbytes
                ),
                "lightzero_array_ceiling_replay_payload_d2h_bytes": float(
                    search_result.visit_policy.nbytes
                    + search_result.root_value.nbytes
                    + (
                        0
                        if search_result.raw_visit_counts is None
                        else search_result.raw_visit_counts.nbytes
                    )
                ),
                "lightzero_array_ceiling_root_observation_copy_bytes": 0.0,
                "lightzero_array_ceiling_python_rows_materialized": 0.0,
                "lightzero_array_ceiling_rnd_materialized_rows": 0.0,
                "lightzero_array_ceiling_resident_obs_reused": 0.0,
                "lightzero_array_ceiling_compact_output_bytes": float(compact_output_bytes),
                "lightzero_array_ceiling_active_roots": float(search_result.root_index.size),
                "lightzero_array_ceiling_zero_mask_roots": float(
                    root_batch.legal_mask.shape[0] - search_result.root_index.size
                ),
                "lightzero_array_ceiling_requested_simulations": float(self._num_simulations),
                "lightzero_array_ceiling_actual_search_simulations": 0.0,
                "lightzero_array_ceiling_recurrent_inference_calls": 0.0,
                "lightzero_array_ceiling_real_ctree_calls": 0.0,
                "lightzero_array_ceiling_illegal_action_count": 0.0,
                "lightzero_array_ceiling_action_checksum": float(
                    np.asarray(search_result.selected_action, dtype=np.int64).sum()
                ),
                "lightzero_array_ceiling_value_checksum": float(
                    np.asarray(search_result.root_value, dtype=np.float64).sum()
                ),
                "lightzero_array_ceiling_visit_shape": list(search_result.visit_policy.shape),
                "lightzero_array_ceiling_compact_search_arrays_stored": 1.0,
                "lightzero_array_ceiling_policy_device": "none",
                "lightzero_array_ceiling_policy_class": str(
                    self._policy_metadata.get("policy_class", type(self._policy).__name__)
                ),
                "lightzero_array_ceiling_model_class": str(
                    self._policy_metadata.get(
                        "model_class",
                        type(getattr(self._policy, "_model", self._policy)).__name__,
                    )
                ),
                "lightzero_array_ceiling_policy_surface": dict(
                    self._policy_metadata.get("surface", {})
                ),
                "lightzero_array_ceiling_first_actions": search_result.selected_action[
                    : min(16, len(search_result.selected_action))
                ]
                .astype(int)
                .tolist(),
                "fixed_shape_search_owner_contract_v1_enabled": True,
                "fixed_shape_search_owner_service_impl": FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL,
                "fixed_shape_search_owner_actual_mcts_simulations": 0.0,
                "fixed_shape_search_owner_output_d2h_bytes": owner_d2h_bytes,
                "fixed_shape_search_owner_preallocated_buffer_bytes": owner_preallocated_bytes,
                **owner_profile,
                "compact_service_contract_v1_enabled": True,
                "compact_service_contract_v1_validation_sec": 0.0,
                "compact_service_contract_v1_contract_id": str(
                    root_batch.metadata.get("contract_id")
                ),
                "compact_service_root_batch_schema_id": str(root_batch.metadata.get("schema_id")),
                "compact_service_search_result_schema_id": str(
                    search_result.metadata.get("schema_id")
                ),
                "compact_service_root_count": float(root_batch.legal_mask.shape[0]),
                "compact_service_active_root_count": float(search_result.root_index.size),
                "compact_service_selected_action_checksum": float(
                    np.asarray(search_result.selected_action, dtype=np.int64).sum()
                ),
                "compact_service_visit_policy_checksum": float(
                    np.asarray(search_result.visit_policy, dtype=np.float64).sum()
                ),
                "compact_service_search_impl": FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL,
                "compact_service_profile_mode": self._mode,
            }
        )

    def _prepare_observation_tensor(
        self,
        *,
        np: Any,
        torch: Any,
        flat_stack: Any,
        stack_dtype: Any,
        device: Any,
    ) -> tuple[Any, float, float, dict[str, float]]:
        telemetry = {
            "pin_memory_sec": 0.0,
            "host_prenormalize_sec": 0.0,
            "resident_first_fill_sec": 0.0,
            "resident_reused": 0.0,
        }

        if self._input_mode == LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_FLOAT32:
            started = time.perf_counter()
            if stack_dtype == np.uint8:
                prepared_stack = flat_stack.astype(np.float32, copy=False) * np.float32(1.0 / 255.0)
            else:
                prepared_stack = flat_stack.astype(np.float32, copy=False)
            telemetry["host_prenormalize_sec"] = time.perf_counter() - started

            started = time.perf_counter()
            obs_tensor = torch.as_tensor(prepared_stack, dtype=torch.float32, device=device)
            _sync_torch_device_if_cuda(torch=torch, device=device)
            h2d_sec = time.perf_counter() - started
            return obs_tensor, h2d_sec, 0.0, telemetry

        if self._input_mode == LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8_PINNED:
            started = time.perf_counter()
            cpu_dtype = getattr(torch, "uint8", None) if stack_dtype == np.uint8 else torch.float32
            if cpu_dtype is None:
                cpu_tensor = torch.as_tensor(flat_stack)
            else:
                cpu_tensor = torch.as_tensor(flat_stack, dtype=cpu_dtype)
            if str(device).startswith("cuda") and hasattr(cpu_tensor, "pin_memory"):
                pin_started = time.perf_counter()
                cpu_tensor = cpu_tensor.pin_memory()
                telemetry["pin_memory_sec"] = time.perf_counter() - pin_started
            if hasattr(cpu_tensor, "to"):
                obs_tensor = cpu_tensor.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=str(device).startswith("cuda"),
                )
            else:
                obs_tensor = torch.as_tensor(flat_stack, dtype=torch.float32, device=device)
            _sync_torch_device_if_cuda(torch=torch, device=device)
            h2d_sec = time.perf_counter() - started

            started = time.perf_counter()
            normalize_sec = 0.0
            if stack_dtype == np.uint8:
                obs_tensor = obs_tensor * (1.0 / 255.0)
                _sync_torch_device_if_cuda(torch=torch, device=device)
                normalize_sec = time.perf_counter() - started
            return obs_tensor, h2d_sec, normalize_sec, telemetry

        if self._input_mode == LIGHTZERO_ARRAY_CEILING_INPUT_MODE_RESIDENT_TORCH_REUSE:
            signature = (
                tuple(int(dim) for dim in getattr(flat_stack, "shape", ())),
                str(device),
                str(stack_dtype),
            )
            if self._resident_obs_tensor is not None and self._resident_obs_signature == signature:
                telemetry["resident_reused"] = 1.0
                return self._resident_obs_tensor, 0.0, 0.0, telemetry

            started = time.perf_counter()
            obs_tensor = torch.as_tensor(flat_stack, dtype=torch.float32, device=device)
            _sync_torch_device_if_cuda(torch=torch, device=device)
            h2d_sec = time.perf_counter() - started

            started = time.perf_counter()
            normalize_sec = 0.0
            if stack_dtype == np.uint8:
                obs_tensor = obs_tensor * (1.0 / 255.0)
                _sync_torch_device_if_cuda(torch=torch, device=device)
                normalize_sec = time.perf_counter() - started
            self._resident_obs_tensor = obs_tensor
            self._resident_obs_signature = signature
            telemetry["resident_first_fill_sec"] = h2d_sec + normalize_sec
            return obs_tensor, h2d_sec, normalize_sec, telemetry

        started = time.perf_counter()
        obs_tensor = torch.as_tensor(flat_stack, dtype=torch.float32, device=device)
        _sync_torch_device_if_cuda(torch=torch, device=device)
        h2d_sec = time.perf_counter() - started

        started = time.perf_counter()
        normalize_sec = 0.0
        if stack_dtype == np.uint8:
            obs_tensor = obs_tensor * (1.0 / 255.0)
            _sync_torch_device_if_cuda(torch=torch, device=device)
            normalize_sec = time.perf_counter() - started
        return obs_tensor, h2d_sec, normalize_sec, telemetry

    def _run_dense_torch_mcts(
        self,
        *,
        torch: Any,
        model: Any,
        root_output: Any,
        root_policy_logits: Any,
        root_value: Any,
        root_latent_state: Any,
        flat_mask: Any,
        active_root_count: int,
        all_roots_legal_fast_path: bool,
        all_actions_legal_fast_path: bool,
        device: Any,
    ) -> dict[str, Any]:
        """Profile-only dense PUCT search with fixed CurvyTron action count.

        This is not a LightZero trainer replacement. It is an Amdahl probe that
        keeps the hot search tensors on the model device so we can price the
        CPU CTree/list boundary.
        """
        if root_latent_state is None:
            root_latent_state = _network_output_field(
                root_output,
                ("latent_state", "hidden_state"),
            )
        if root_latent_state is None:
            raise ValueError("dense_torch_mcts requires latent_state from initial_inference")
        if not _is_torch_like_tensor(root_latent_state):
            raise ValueError("dense_torch_mcts requires a torch-like latent tensor")

        cfg = getattr(self._policy, "_cfg", SimpleNamespace())
        pb_c_base = float(getattr(cfg, "pb_c_base", 19652))
        pb_c_init = float(getattr(cfg, "pb_c_init", 1.25))
        discount_factor = float(getattr(cfg, "discount_factor", 0.997))
        root_dirichlet_alpha = float(getattr(cfg, "root_dirichlet_alpha", 0.3))
        root_noise_weight = float(getattr(cfg, "root_noise_weight", 0.0))
        value_delta_max = float(getattr(cfg, "value_delta_max", 0.01))
        num_simulations = int(self._num_simulations)
        root_count = int(active_root_count)
        max_nodes = num_simulations + 1
        row_index = torch.arange(root_count, dtype=torch.long, device=device)

        flat_mask_tensor = torch.as_tensor(flat_mask > 0.0, dtype=torch.bool, device=device)
        root_logits = root_policy_logits.reshape(root_count, -1)[:, :ACTION_COUNT].float()
        root_priors = torch.softmax(
            root_logits.masked_fill(~flat_mask_tensor, -1.0e9),
            dim=1,
        )
        if root_noise_weight > 0.0:
            alpha = torch.full(
                (ACTION_COUNT,),
                root_dirichlet_alpha,
                dtype=torch.float32,
                device=device,
            )
            noise = torch.distributions.Dirichlet(alpha).sample((root_count,))
            noise = noise * flat_mask_tensor.float()
            noise = noise / noise.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            root_priors = root_priors * (1.0 - root_noise_weight) + noise * root_noise_weight
            root_priors = root_priors / root_priors.sum(dim=1, keepdim=True).clamp_min(1.0e-12)

        edge_child = torch.full(
            (root_count, max_nodes, ACTION_COUNT),
            -1,
            dtype=torch.long,
            device=device,
        )
        edge_visit = torch.zeros(
            (root_count, max_nodes, ACTION_COUNT),
            dtype=torch.float32,
            device=device,
        )
        edge_value_sum = torch.zeros_like(edge_visit)
        edge_reward = torch.zeros_like(edge_visit)
        edge_prior = torch.zeros_like(edge_visit)
        edge_prior[:, 0, :] = root_priors

        latent_pool = torch.empty(
            (max_nodes, root_count) + tuple(int(dim) for dim in root_latent_state.shape[1:]),
            dtype=root_latent_state.dtype,
            device=device,
        )
        latent_pool[0].copy_(root_latent_state)
        node_latent_slot = torch.zeros(
            (root_count, max_nodes),
            dtype=torch.long,
            device=device,
        )
        next_node_index = torch.ones((root_count,), dtype=torch.long, device=device)
        min_value = torch.full((root_count,), math.inf, dtype=torch.float32, device=device)
        max_value = torch.full((root_count,), -math.inf, dtype=torch.float32, device=device)

        recurrent_inference_sec = 0.0
        search_update_sec = 0.0
        output_assembly_sec = 0.0
        action_shape_mode = "none"

        root_value_tensor = _policy_inverse_scalar_value(
            policy=self._policy,
            value=root_value,
            torch=torch,
            root_count=root_count,
            device=device,
        )
        path_node_history = torch.empty(
            (num_simulations, root_count),
            dtype=torch.long,
            device=device,
        )
        path_action_history = torch.empty_like(path_node_history)
        path_active_history = torch.empty(
            (num_simulations, root_count),
            dtype=torch.bool,
            device=device,
        )
        (
            select_helper,
            backup_helper,
            compile_telemetry,
        ) = self._dense_torch_mcts_compile_helpers(
            torch=torch,
            device=device,
            edge_child=edge_child,
            edge_visit=edge_visit,
            edge_value_sum=edge_value_sum,
            edge_reward=edge_reward,
            edge_prior=edge_prior,
            latent_pool=latent_pool,
            node_latent_slot=node_latent_slot,
            next_node_index=next_node_index,
            min_value=min_value,
            max_value=max_value,
            path_node_history=path_node_history,
            path_action_history=path_action_history,
            path_active_history=path_active_history,
            flat_mask_tensor=flat_mask_tensor,
            row_index=row_index,
            root_latent_state=root_latent_state,
            root_noise_weight=root_noise_weight,
            num_simulations=num_simulations,
            root_count=root_count,
            all_roots_legal_fast_path=all_roots_legal_fast_path,
            all_actions_legal_fast_path=all_actions_legal_fast_path,
            pb_c_base=pb_c_base,
            pb_c_init=pb_c_init,
            discount_factor=discount_factor,
            value_delta_max=value_delta_max,
        )

        for simulation_index in range(num_simulations):
            leaf_parent, leaf_action = select_helper(
                edge_child,
                edge_visit,
                edge_value_sum,
                edge_prior,
                path_node_history,
                path_action_history,
                path_active_history,
                flat_mask_tensor,
                row_index,
                min_value,
                max_value,
                simulation_index,
                pb_c_base,
                pb_c_init,
                value_delta_max,
            )

            parent_latent_slots = node_latent_slot[row_index, leaf_parent]
            parent_latents = latent_pool[parent_latent_slots, row_index]
            action_input = _recurrent_action_input(
                np=None,
                torch=torch,
                actions=leaf_action,
                device=device,
                mode=action_shape_mode,
            )
            started = time.perf_counter()
            with torch.no_grad():
                try:
                    recurrent_output = model.recurrent_inference(parent_latents, action_input)
                    if action_shape_mode == "none":
                        action_shape_mode = "flat"
                except Exception:
                    action_input = _recurrent_action_input(
                        np=None,
                        torch=torch,
                        actions=leaf_action,
                        device=device,
                        mode="column",
                    )
                    recurrent_output = model.recurrent_inference(parent_latents, action_input)
                    action_shape_mode = "column"
            recurrent_inference_sec += time.perf_counter() - started

            started = time.perf_counter()
            next_latent_state = _network_output_field(
                recurrent_output,
                ("latent_state", "hidden_state"),
            )
            if next_latent_state is None:
                raise ValueError("dense_torch_mcts requires recurrent latent_state")
            reward = _policy_inverse_scalar_value(
                policy=self._policy,
                value=_network_output_field(recurrent_output, ("reward", "value_prefix")),
                torch=torch,
                root_count=root_count,
                device=device,
            )
            value = _policy_inverse_scalar_value(
                policy=self._policy,
                value=_network_output_field(recurrent_output, ("value", "predicted_value")),
                torch=torch,
                root_count=root_count,
                device=device,
            )
            recurrent_logits = _network_output_field(
                recurrent_output,
                ("policy_logits", "policy"),
            )
            if recurrent_logits is None:
                raise ValueError("dense_torch_mcts requires recurrent policy_logits")
            recurrent_priors = torch.softmax(
                recurrent_logits.reshape(root_count, -1)[:, :ACTION_COUNT].float(),
                dim=1,
            )
            next_node_index, min_value, max_value = backup_helper(
                edge_child,
                edge_visit,
                edge_value_sum,
                edge_reward,
                edge_prior,
                latent_pool,
                node_latent_slot,
                next_node_index,
                min_value,
                max_value,
                path_node_history,
                path_action_history,
                path_active_history,
                row_index,
                leaf_parent,
                leaf_action,
                next_latent_state,
                reward,
                value,
                recurrent_priors,
                simulation_index,
                discount_factor,
            )
            search_update_sec += time.perf_counter() - started

        started = time.perf_counter()
        root_visits = edge_visit[:, 0, :]
        root_visits = root_visits * flat_mask_tensor.float()
        raw_visit_totals = root_visits.sum(dim=1, keepdim=True)
        visit_totals = raw_visit_totals.clamp_min(1.0)
        visit_distributions = root_visits / visit_totals
        actions = torch.argmax(root_visits.masked_fill(~flat_mask_tensor, -1.0e9), dim=1)
        root_values = edge_value_sum[:, 0, :].sum(dim=1) / visit_totals.reshape(-1)
        root_values = torch.where(
            raw_visit_totals.reshape(-1) > 0.0,
            root_values,
            root_value_tensor,
        )
        _sync_torch_device_if_cuda(torch=torch, device=device)
        output_assembly_sec += time.perf_counter() - started

        return {
            "actions": actions,
            "root_values": root_values,
            "visit_distributions": visit_distributions,
            "recurrent_inference_sec": recurrent_inference_sec,
            "search_update_sec": search_update_sec,
            "output_assembly_sec": output_assembly_sec,
            "recurrent_calls": num_simulations,
            "action_shape_mode": action_shape_mode,
            "compile_telemetry": compile_telemetry,
        }

    def _dense_torch_mcts_compile_helpers(
        self,
        *,
        torch: Any,
        device: Any,
        edge_child: Any,
        edge_visit: Any,
        edge_value_sum: Any,
        edge_reward: Any,
        edge_prior: Any,
        latent_pool: Any,
        node_latent_slot: Any,
        next_node_index: Any,
        min_value: Any,
        max_value: Any,
        path_node_history: Any,
        path_action_history: Any,
        path_active_history: Any,
        flat_mask_tensor: Any,
        row_index: Any,
        root_latent_state: Any,
        root_noise_weight: float,
        num_simulations: int,
        root_count: int,
        all_roots_legal_fast_path: bool,
        all_actions_legal_fast_path: bool,
        pb_c_base: float,
        pb_c_init: float,
        discount_factor: float,
        value_delta_max: float,
    ) -> tuple[Any, Any, dict[str, Any]]:
        eager_select = _dense_torch_mcts_select_leaf_fixed(torch=torch)
        eager_backup = _dense_torch_mcts_expand_and_backup_fixed(torch=torch)
        telemetry = _dense_torch_mcts_compile_default_telemetry(self._mode)
        if self._mode != LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE:
            return eager_select, eager_backup, telemetry

        telemetry.update(
            {
                "lightzero_array_ceiling_compile_status": "fallback_precondition",
                "lightzero_array_ceiling_compile_attempted": 0.0,
                "lightzero_array_ceiling_compile_enabled": 0.0,
                "lightzero_array_ceiling_compile_reason": "",
                "lightzero_array_ceiling_compile_mode": "reduce-overhead",
                "lightzero_array_ceiling_compile_fullgraph": 1.0,
            }
        )
        if not str(device).startswith("cuda"):
            telemetry["lightzero_array_ceiling_compile_reason"] = "requires_cuda_device"
            return eager_select, eager_backup, telemetry
        cuda = getattr(torch, "cuda", None)
        cuda_available = True
        if cuda is not None and hasattr(cuda, "is_available"):
            try:
                cuda_available = bool(cuda.is_available())
            except Exception:
                cuda_available = False
        if not cuda_available:
            telemetry["lightzero_array_ceiling_compile_reason"] = "torch_cuda_unavailable"
            return eager_select, eager_backup, telemetry
        if not hasattr(torch, "compile"):
            telemetry["lightzero_array_ceiling_compile_reason"] = "torch_compile_unavailable"
            return eager_select, eager_backup, telemetry
        if not all_roots_legal_fast_path:
            telemetry["lightzero_array_ceiling_compile_reason"] = "requires_all_roots_legal"
            return eager_select, eager_backup, telemetry
        if not all_actions_legal_fast_path:
            telemetry["lightzero_array_ceiling_compile_reason"] = "requires_all_actions_legal"
            return eager_select, eager_backup, telemetry
        if float(root_noise_weight) != 0.0:
            telemetry["lightzero_array_ceiling_compile_reason"] = "requires_root_noise_zero"
            return eager_select, eager_backup, telemetry

        signature = (
            int(root_count),
            int(num_simulations),
            tuple(int(dim) for dim in getattr(root_latent_state, "shape", ())),
            str(getattr(root_latent_state, "dtype", "")),
            str(getattr(root_latent_state, "device", device)),
            ACTION_COUNT,
        )
        telemetry["lightzero_array_ceiling_compile_signature"] = list(signature[:2]) + [
            list(signature[2]),
            signature[3],
            signature[4],
            signature[5],
        ]
        if (
            self._dense_compile_select_helper is not None
            and self._dense_compile_backup_helper is not None
            and self._dense_compile_signature == signature
        ):
            telemetry.update(
                {
                    "lightzero_array_ceiling_compile_status": "compiled_cached",
                    "lightzero_array_ceiling_compile_enabled": 1.0,
                    "lightzero_array_ceiling_compile_reason": "cached",
                }
            )
            return (
                self._dense_compile_select_helper,
                self._dense_compile_backup_helper,
                telemetry,
            )

        started = time.perf_counter()
        telemetry["lightzero_array_ceiling_compile_attempted"] = 1.0
        try:
            compiled_select = torch.compile(
                eager_select,
                mode="reduce-overhead",
                fullgraph=True,
            )
            compiled_backup = torch.compile(
                eager_backup,
                mode="reduce-overhead",
                fullgraph=True,
            )
            warmup_calls = _warm_dense_torch_mcts_compiled_helpers(
                select_helper=compiled_select,
                backup_helper=compiled_backup,
                edge_child=edge_child,
                edge_visit=edge_visit,
                edge_value_sum=edge_value_sum,
                edge_reward=edge_reward,
                edge_prior=edge_prior,
                latent_pool=latent_pool,
                node_latent_slot=node_latent_slot,
                next_node_index=next_node_index,
                min_value=min_value,
                max_value=max_value,
                path_node_history=path_node_history,
                path_action_history=path_action_history,
                path_active_history=path_active_history,
                flat_mask_tensor=flat_mask_tensor,
                row_index=row_index,
                root_latent_state=root_latent_state,
                num_simulations=num_simulations,
                root_count=root_count,
                pb_c_base=pb_c_base,
                pb_c_init=pb_c_init,
                discount_factor=discount_factor,
                value_delta_max=value_delta_max,
            )
            _sync_torch_device_if_cuda(torch=torch, device=device)
        except Exception as exc:
            self._dense_compile_select_helper = None
            self._dense_compile_backup_helper = None
            self._dense_compile_signature = None
            telemetry.update(
                {
                    "lightzero_array_ceiling_compile_status": "fallback_compile_failed",
                    "lightzero_array_ceiling_compile_enabled": 0.0,
                    "lightzero_array_ceiling_compile_reason": (f"{type(exc).__name__}: {exc}"),
                    "lightzero_array_ceiling_compile_warmup_sec": (time.perf_counter() - started),
                    "lightzero_array_ceiling_compile_warmup_calls": 0.0,
                }
            )
            return eager_select, eager_backup, telemetry

        self._dense_compile_select_helper = compiled_select
        self._dense_compile_backup_helper = compiled_backup
        self._dense_compile_signature = signature
        telemetry.update(
            {
                "lightzero_array_ceiling_compile_status": "compiled",
                "lightzero_array_ceiling_compile_enabled": 1.0,
                "lightzero_array_ceiling_compile_reason": "warmup_succeeded",
                "lightzero_array_ceiling_compile_warmup_sec": (time.perf_counter() - started),
                "lightzero_array_ceiling_compile_warmup_calls": float(warmup_calls),
            }
        )
        return compiled_select, compiled_backup, telemetry

    def _zero_root_telemetry(
        self,
        *,
        stack: Any,
        mask: Any,
        tensor_prepare_sec: float,
        total_root_count: int,
        dropped_zero_mask_root_count: int,
        device: Any,
    ) -> dict[str, Any]:
        return {
            "host_to_device_sec": 0.0,
            "host_to_device_bytes": 0.0,
            "normalize_sec": 0.0,
            "device_sec": 0.0,
            "readback_sec": 0.0,
            "total_sec": tensor_prepare_sec,
            "simulations": 0.0,
            "channels": 0.0,
            "roots": 0.0,
            "input_rank": float(len(getattr(stack, "shape", ()))),
            "input_bytes": float(getattr(stack, "nbytes", 0)),
            "model_eval_count": 0.0,
            "output_readback_bytes": 0.0,
            "action_mask_consumed": 1.0,
            "compile_excluded_by_warmup": 1.0,
            "lightzero_array_ceiling_mode": self._mode,
            "lightzero_array_ceiling_semantics": self.semantics,
            "lightzero_array_ceiling_input_mode": self._input_mode,
            **(
                {
                    **_dense_torch_mcts_compile_default_telemetry(self._mode),
                    "lightzero_array_ceiling_compile_status": "not_attempted_zero_roots",
                    "lightzero_array_ceiling_compile_reason": "no_active_roots",
                }
                if self._mode == LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE
                else _dense_torch_mcts_compile_default_telemetry(self._mode)
            ),
            "lightzero_array_ceiling_total_sec": tensor_prepare_sec,
            "lightzero_array_ceiling_tensor_prepare_sec": tensor_prepare_sec,
            "lightzero_array_ceiling_h2d_sec": 0.0,
            "lightzero_array_ceiling_normalize_sec": 0.0,
            "lightzero_array_ceiling_pin_memory_sec": 0.0,
            "lightzero_array_ceiling_host_prenormalize_sec": 0.0,
            "lightzero_array_ceiling_resident_first_fill_sec": 0.0,
            "lightzero_array_ceiling_resident_reused": 0.0,
            "lightzero_array_ceiling_initial_inference_sec": 0.0,
            "lightzero_array_ceiling_initial_inference_calls": 0.0,
            "lightzero_array_ceiling_recurrent_inference_sec": 0.0,
            "lightzero_array_ceiling_recurrent_inference_calls": 0.0,
            "lightzero_array_ceiling_search_update_sec": 0.0,
            "lightzero_array_ceiling_output_assembly_sec": 0.0,
            "lightzero_array_ceiling_readback_sec": 0.0,
            "lightzero_array_ceiling_output_bytes": 0.0,
            "lightzero_array_ceiling_policy_shape": [],
            "lightzero_array_ceiling_value_shape": [],
            "lightzero_array_ceiling_action_shape": [],
            "lightzero_array_ceiling_action_checksum": 0.0,
            "lightzero_array_ceiling_value_checksum": 0.0,
            "lightzero_array_ceiling_illegal_action_count": 0.0,
            "lightzero_array_ceiling_action_shape_mode": "none",
            "lightzero_array_ceiling_policy_device": str(device),
            "lightzero_array_ceiling_policy_class": str(
                self._policy_metadata.get("policy_class", type(self._policy).__name__)
            ),
            "lightzero_total_root_count": float(total_root_count),
            "lightzero_filtered_zero_mask_root_count": float(dropped_zero_mask_root_count),
            "lightzero_root_count": 0.0,
            "lightzero_roots_per_call": 0.0,
            "lightzero_consumer_input_bytes": float(getattr(stack, "nbytes", 0))
            + float(getattr(mask, "nbytes", 0)),
        }


def _dense_torch_mcts_compile_default_telemetry(mode: str) -> dict[str, Any]:
    requested = mode == LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE
    return {
        "lightzero_array_ceiling_compile_status": (
            "not_attempted" if requested else "not_requested"
        ),
        "lightzero_array_ceiling_compile_attempted": 0.0,
        "lightzero_array_ceiling_compile_enabled": 0.0,
        "lightzero_array_ceiling_compile_reason": "",
        "lightzero_array_ceiling_compile_mode": ("reduce-overhead" if requested else "none"),
        "lightzero_array_ceiling_compile_fullgraph": 0.0,
        "lightzero_array_ceiling_compile_warmup_sec": 0.0,
        "lightzero_array_ceiling_compile_warmup_calls": 0.0,
        "lightzero_array_ceiling_compile_signature": [],
        "lightzero_array_ceiling_compile_helper": (
            "select_leaf+expand_backup" if requested else "none"
        ),
    }


def _dense_torch_mcts_select_leaf_fixed(*, torch: Any) -> Any:
    def select_leaf(
        edge_child: Any,
        edge_visit: Any,
        edge_value_sum: Any,
        edge_prior: Any,
        path_node_history: Any,
        path_action_history: Any,
        path_active_history: Any,
        flat_mask_tensor: Any,
        row_index: Any,
        min_value: Any,
        max_value: Any,
        simulation_index: int,
        pb_c_base: float,
        pb_c_init: float,
        value_delta_max: float,
    ) -> tuple[Any, Any]:
        current_node = torch.zeros_like(row_index)
        active = torch.ones(
            row_index.shape,
            dtype=torch.bool,
            device=row_index.device,
        )
        leaf_parent = torch.zeros_like(row_index)
        leaf_action = torch.zeros_like(row_index)

        for depth_index in range(simulation_index + 1):
            visits = edge_visit[row_index, current_node, :]
            values = edge_value_sum[row_index, current_node, :]
            priors = edge_prior[row_index, current_node, :]
            parent_visit = visits.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_value = values / visits.clamp_min(1.0)
            unvisited = visits <= 0.0
            normalized_value = (mean_value - min_value[:, None]) / (
                (max_value[:, None] - min_value[:, None]).clamp_min(value_delta_max)
            )
            normalized_value = torch.where(
                unvisited,
                torch.zeros_like(normalized_value),
                normalized_value,
            )
            pb_c = (
                (torch.log((parent_visit + pb_c_base + 1.0) / pb_c_base) + pb_c_init)
                * torch.sqrt(parent_visit)
                / (visits + 1.0)
            )
            score = normalized_value + priors * pb_c
            if depth_index == 0:
                score = score.masked_fill(~flat_mask_tensor, -1.0e9)
            selected_action = torch.argmax(score, dim=1)

            path_node_history[depth_index].copy_(current_node)
            path_action_history[depth_index].copy_(selected_action)
            path_active_history[depth_index].copy_(active)

            selected_child = edge_child[row_index, current_node, selected_action]
            needs_expand = active & (selected_child < 0)
            leaf_parent = torch.where(needs_expand, current_node, leaf_parent)
            leaf_action = torch.where(needs_expand, selected_action, leaf_action)
            continue_active = active & (selected_child >= 0)
            current_node = torch.where(continue_active, selected_child, current_node)
            active = continue_active

        return leaf_parent, leaf_action

    return select_leaf


def _dense_torch_mcts_expand_and_backup_fixed(*, torch: Any) -> Any:
    def expand_and_backup(
        edge_child: Any,
        edge_visit: Any,
        edge_value_sum: Any,
        edge_reward: Any,
        edge_prior: Any,
        latent_pool: Any,
        node_latent_slot: Any,
        next_node_index: Any,
        min_value: Any,
        max_value: Any,
        path_node_history: Any,
        path_action_history: Any,
        path_active_history: Any,
        row_index: Any,
        leaf_parent: Any,
        leaf_action: Any,
        next_latent_state: Any,
        reward: Any,
        value: Any,
        recurrent_priors: Any,
        simulation_index: int,
        discount_factor: float,
    ) -> tuple[Any, Any, Any]:
        new_node = next_node_index
        edge_child[row_index, leaf_parent, leaf_action] = new_node
        edge_reward[row_index, leaf_parent, leaf_action] = reward
        edge_prior[row_index, new_node, :] = recurrent_priors
        latent_pool[simulation_index + 1].copy_(next_latent_state)
        node_latent_slot[row_index, new_node] = simulation_index + 1
        next_node_index = next_node_index + 1

        bootstrap = value
        for depth_index in range(simulation_index, -1, -1):
            nodes = path_node_history[depth_index]
            actions = path_action_history[depth_index]
            mask = path_active_history[depth_index]
            mask_f = mask.float()
            backed_value = edge_reward[row_index, nodes, actions] + discount_factor * bootstrap
            edge_visit[row_index, nodes, actions] += mask_f
            edge_value_sum[row_index, nodes, actions] += backed_value * mask_f
            min_value = torch.where(
                mask,
                torch.minimum(min_value, backed_value),
                min_value,
            )
            max_value = torch.where(
                mask,
                torch.maximum(max_value, backed_value),
                max_value,
            )
            bootstrap = torch.where(mask, backed_value, bootstrap)
        return next_node_index, min_value, max_value

    return expand_and_backup


def _warm_dense_torch_mcts_compiled_helpers(
    *,
    select_helper: Any,
    backup_helper: Any,
    edge_child: Any,
    edge_visit: Any,
    edge_value_sum: Any,
    edge_reward: Any,
    edge_prior: Any,
    latent_pool: Any,
    node_latent_slot: Any,
    next_node_index: Any,
    min_value: Any,
    max_value: Any,
    path_node_history: Any,
    path_action_history: Any,
    path_active_history: Any,
    flat_mask_tensor: Any,
    row_index: Any,
    root_latent_state: Any,
    num_simulations: int,
    root_count: int,
    pb_c_base: float,
    pb_c_init: float,
    discount_factor: float,
    value_delta_max: float,
) -> int:
    warm_edge_child = edge_child.clone()
    warm_edge_visit = edge_visit.clone()
    warm_edge_value_sum = edge_value_sum.clone()
    warm_edge_reward = edge_reward.clone()
    warm_edge_prior = edge_prior.clone()
    warm_latent_pool = latent_pool.clone()
    warm_node_latent_slot = node_latent_slot.clone()
    warm_next_node_index = next_node_index.clone()
    warm_min_value = min_value.clone()
    warm_max_value = max_value.clone()
    warm_path_node_history = path_node_history.clone()
    warm_path_action_history = path_action_history.clone()
    warm_path_active_history = path_active_history.clone()
    recurrent_reward = edge_visit.new_zeros((int(root_count),))
    recurrent_value = edge_visit.new_zeros((int(root_count),))
    recurrent_priors = edge_visit.new_full(
        (int(root_count), ACTION_COUNT),
        1.0 / float(ACTION_COUNT),
    )
    next_latent_state = root_latent_state.clone()

    calls = 0
    for simulation_index in range(int(num_simulations)):
        leaf_parent, leaf_action = select_helper(
            warm_edge_child,
            warm_edge_visit,
            warm_edge_value_sum,
            warm_edge_prior,
            warm_path_node_history,
            warm_path_action_history,
            warm_path_active_history,
            flat_mask_tensor,
            row_index,
            warm_min_value,
            warm_max_value,
            simulation_index,
            pb_c_base,
            pb_c_init,
            value_delta_max,
        )
        calls += 1
        warm_next_node_index, warm_min_value, warm_max_value = backup_helper(
            warm_edge_child,
            warm_edge_visit,
            warm_edge_value_sum,
            warm_edge_reward,
            warm_edge_prior,
            warm_latent_pool,
            warm_node_latent_slot,
            warm_next_node_index,
            warm_min_value,
            warm_max_value,
            warm_path_node_history,
            warm_path_action_history,
            warm_path_active_history,
            row_index,
            leaf_parent,
            leaf_action,
            next_latent_state,
            recurrent_reward,
            recurrent_value,
            recurrent_priors,
            simulation_index,
            discount_factor,
        )
        calls += 1
    return calls


def _sync_torch_device_if_cuda(*, torch: Any, device: Any) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        return


def _network_output_field(output: Any, names: tuple[str, ...]) -> Any:
    if isinstance(output, Mapping):
        for name in names:
            if name in output:
                return output[name]
    for name in names:
        if hasattr(output, name):
            return getattr(output, name)
    return None


def _is_torch_like_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu")


def _masked_policy_arrays(
    *,
    np: Any,
    torch: Any,
    policy_logits: Any,
    value: Any,
    flat_mask: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    if _is_torch_like_tensor(policy_logits):
        logits = policy_logits.reshape(int(flat_mask.shape[0]), -1)[:, :ACTION_COUNT]
        mask_tensor = torch.as_tensor(flat_mask > 0.0, device=device)
        masked_logits = logits.masked_fill(mask_tensor <= 0, -1.0e9)
        priors = torch.softmax(masked_logits, dim=1)
        actions = torch.argmax(masked_logits, dim=1)
        if value is None:
            values = torch.zeros((int(flat_mask.shape[0]),), dtype=torch.float32, device=device)
        else:
            values = value.reshape(int(flat_mask.shape[0]), -1)[:, 0].float()
        return actions, values, priors

    logits_np = np.asarray(policy_logits, dtype=np.float32).reshape(int(flat_mask.shape[0]), -1)[
        :, :ACTION_COUNT
    ]
    masked = np.where(np.asarray(flat_mask) > 0.0, logits_np, -1.0e9)
    exp = np.exp(masked - np.max(masked, axis=1, keepdims=True))
    priors = exp / np.maximum(exp.sum(axis=1, keepdims=True), np.float32(1.0e-12))
    actions = np.argmax(masked, axis=1).astype(np.int64)
    if value is None:
        values = np.zeros((int(flat_mask.shape[0]),), dtype=np.float32)
    else:
        values = np.asarray(value, dtype=np.float32).reshape(int(flat_mask.shape[0]), -1)[:, 0]
    return actions, values, priors


def _policy_inverse_scalar_value(
    *,
    policy: Any,
    value: Any,
    torch: Any,
    root_count: int,
    device: Any,
) -> Any:
    if value is None:
        return torch.zeros((int(root_count),), dtype=torch.float32, device=device)
    transformed = value
    inverse = getattr(policy, "inverse_scalar_transform_handle", None)
    if inverse is not None:
        try:
            transformed = inverse(value)
        except Exception:
            transformed = value
    if _is_torch_like_tensor(transformed):
        return transformed.reshape(int(root_count), -1)[:, 0].float().to(device=device)
    try:
        return torch.as_tensor(transformed, dtype=torch.float32, device=device).reshape(
            int(root_count),
            -1,
        )[:, 0]
    except Exception:
        return torch.zeros((int(root_count),), dtype=torch.float32, device=device)


def _zeros_like_policy_array(*, np: Any, torch: Any, root_count: int, device: Any) -> Any:
    if hasattr(torch, "zeros"):
        try:
            return torch.zeros((int(root_count), ACTION_COUNT), dtype=torch.float32, device=device)
        except Exception:
            pass
    return np.zeros((int(root_count), ACTION_COUNT), dtype=np.float32)


def _zeros_like_value_array(*, np: Any, torch: Any, root_count: int, device: Any) -> Any:
    if hasattr(torch, "zeros"):
        try:
            return torch.zeros((int(root_count),), dtype=torch.float32, device=device)
        except Exception:
            pass
    return np.zeros((int(root_count),), dtype=np.float32)


def _recurrent_action_input(
    *,
    np: Any,
    torch: Any,
    actions: Any,
    device: Any,
    mode: str,
) -> Any:
    if _is_torch_like_tensor(actions):
        action_tensor = actions.long()
        if mode == "column":
            return action_tensor.reshape(-1, 1)
        return action_tensor.reshape(-1)
    action_np = np.asarray(actions, dtype=np.int64)
    if mode == "column":
        action_np = action_np.reshape(-1, 1)
    else:
        action_np = action_np.reshape(-1)
    try:
        return torch.as_tensor(action_np, device=device)
    except Exception:
        return action_np


def _scatter_visit_and_value(
    *,
    np: Any,
    torch: Any,
    visits: Any,
    value_accum: Any,
    actions: Any,
    values: Any,
) -> None:
    if _is_torch_like_tensor(visits):
        index = actions.long().reshape(-1, 1)
        visits.scatter_add_(
            1,
            index,
            torch.ones(index.shape, dtype=torch.float32, device=visits.device),
        )
        value_accum.add_(values.float().reshape(-1))
        return
    actions_np = np.asarray(actions, dtype=np.int64).reshape(-1)
    np.add.at(visits, (np.arange(actions_np.size), actions_np), 1.0)
    value_accum += np.asarray(values, dtype=np.float32).reshape(-1)


def _array_to_numpy(*, np: Any, value: Any) -> Any:
    if _is_torch_like_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _network_output_shape_summary(output: Any) -> dict[str, Any]:
    fields = (
        "value",
        "reward",
        "policy_logits",
        "latent_state",
        "hidden_state",
        "value_prefix",
    )
    summary: dict[str, Any] = {"type": type(output).__name__}
    for field in fields:
        if not hasattr(output, field):
            continue
        item = getattr(output, field)
        summary[field] = {
            "shape": [int(dim) for dim in getattr(item, "shape", ())],
            "dtype": str(getattr(item, "dtype", type(item).__name__)),
            "device": str(getattr(item, "device", "")),
        }
    if hasattr(output, "_fields"):
        summary["fields"] = [str(field) for field in getattr(output, "_fields")]
    return summary


def _trail_stats_after_owner_pack(
    *,
    np: Any,
    compact_state: dict[str, Any],
    render_trail_slots: int,
    max_render_trail_slots: int | None = None,
) -> dict[str, float]:
    cursor = np.asarray(compact_state["trail_write_cursor"], dtype=np.float64)
    if cursor.ndim != 1:
        raise ValueError(f"trail_write_cursor must be rank 1, got shape {cursor.shape}")
    slots = float(max(1, int(render_trail_slots)))
    env_slots = float(max(1, int(np.asarray(compact_state["trail_active"]).shape[1])))
    fractions = cursor / slots
    dropped = np.maximum(cursor - slots, 0.0)
    truncated = dropped > 0.0
    return {
        "env_trail_slots": env_slots,
        "max_render_trail_slots": float(
            max(
                1,
                int(
                    max_render_trail_slots
                    if max_render_trail_slots is not None
                    else render_trail_slots
                ),
            )
        ),
        "render_trail_slots": slots,
        "active_trail_count_min": float(cursor.min()) if cursor.size else 0.0,
        "active_trail_count_median": _median_array(np=np, values=cursor),
        "active_trail_count_p95": _p95_array(np=np, values=cursor),
        "active_trail_count_max": float(cursor.max()) if cursor.size else 0.0,
        "active_trail_count_sum": float(cursor.sum()) if cursor.size else 0.0,
        "active_trail_fraction_median": _median_array(np=np, values=fractions),
        "active_trail_fraction_p95": _p95_array(np=np, values=fractions),
        "active_trail_fraction_max": float(fractions.max()) if fractions.size else 0.0,
        "render_truncation_row_count": float(np.count_nonzero(truncated)),
        "render_truncation_row_fraction": float(np.mean(truncated)) if truncated.size else 0.0,
        "render_truncation_max_dropped_slots": float(dropped.max()) if dropped.size else 0.0,
    }


def _select_render_trail_slots(
    *,
    np: Any,
    compact_state: dict[str, Any],
    config: dict[str, Any],
) -> int:
    max_slots = int(config["trail_slots"])
    if not bool(config.get("dynamic_render_trail_slots", False)):
        return max_slots
    cursor = np.asarray(compact_state["trail_write_cursor"], dtype=np.int64)
    max_active = int(cursor.max()) if cursor.size else 1
    min_slots = int(config.get("min_render_trail_slots", DEFAULT_DYNAMIC_MIN_RENDER_TRAIL_SLOTS))
    requested = _ceil_power_of_two(max(1, min_slots, max_active))
    return min(max_slots, requested)


def _assert_no_render_truncation_if_required(
    *,
    config: dict[str, Any],
    trail_stats: dict[str, float],
) -> None:
    if bool(config.get("allow_render_truncation", False)):
        return
    row_count = int(trail_stats["render_truncation_row_count"])
    if row_count == 0:
        return
    max_dropped = int(trail_stats["render_truncation_max_dropped_slots"])
    raise ValueError(
        "render_trail_slots would drop active trails; increase trail_slots, "
        "enable dynamic_render_trail_slots with a higher cap, or pass "
        "allow_render_truncation=true for a deliberate lossy diagnostic row. "
        f"row_count={row_count}, max_dropped_slots={max_dropped}, "
        f"render_trail_slots={int(trail_stats['render_trail_slots'])}, "
        f"max_active_trail_count={int(trail_stats['active_trail_count_max'])}"
    )


def _truncate_compact_trails_for_render(
    *,
    np: Any,
    compact_state: dict[str, Any],
    render_trail_slots: int,
) -> dict[str, Any]:
    slots = int(render_trail_slots)
    if slots < 1:
        raise ValueError(f"render_trail_slots must be positive, got {slots}")
    truncated = dict(compact_state)
    for key in TRAIL_ARRAY_KEYS:
        value = np.asarray(compact_state[key])
        if value.ndim != 2:
            raise ValueError(f"{key} must be rank 2, got shape {value.shape}")
        truncated[key] = value[:, :slots].copy()
    truncated["trail_write_cursor"] = np.clip(
        np.asarray(compact_state["trail_write_cursor"], dtype=np.int32),
        0,
        slots,
    ).astype(np.int32, copy=True)
    return truncated


def _ceil_power_of_two(value: int) -> int:
    checked = int(value)
    if checked <= 1:
        return 1
    return 1 << (checked - 1).bit_length()


def _view_major_to_row_major_frames(view_major: Any, *, batch_size: int) -> Any:
    import numpy as np

    frames = np.asarray(view_major)
    expected = (int(batch_size) * PLAYER_COUNT, 1, TARGET_SIZE, TARGET_SIZE)
    if frames.shape != expected:
        raise ValueError(f"view-major frames must have shape {expected}, got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"view-major frames must be uint8, got {frames.dtype}")
    return (
        frames.reshape(PLAYER_COUNT, int(batch_size), 1, TARGET_SIZE, TARGET_SIZE)
        .transpose(1, 0, 2, 3, 4)
        .copy()
    )


def _push_row_major_frames_into_stack(
    stacks: Any,
    frames: Any,
    *,
    row_mask: Any | None = None,
    reset_selected_rows: bool = False,
) -> float:
    import numpy as np

    started = time.perf_counter()
    stack_array = np.asarray(stacks)
    frame_array = np.asarray(frames)
    if stack_array.dtype != np.float32:
        raise ValueError(f"stacks must be float32, got {stack_array.dtype}")
    expected_stack = (
        frame_array.shape[0],
        PLAYER_COUNT,
        POLICY_FRAME_STACK_DEPTH,
        TARGET_SIZE,
        TARGET_SIZE,
    )
    expected_frames = (frame_array.shape[0], PLAYER_COUNT, 1, TARGET_SIZE, TARGET_SIZE)
    if stack_array.shape != expected_stack:
        raise ValueError(f"stacks must have shape {expected_stack}, got {stack_array.shape}")
    if frame_array.shape != expected_frames:
        raise ValueError(f"frames must have shape {expected_frames}, got {frame_array.shape}")
    if frame_array.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {frame_array.dtype}")
    rows = _stack_row_indices(np=np, batch_size=frame_array.shape[0], row_mask=row_mask)
    if reset_selected_rows:
        stack_array[rows] = 0.0
    stack_array[rows, :, :-1] = stack_array[rows, :, 1:]
    stack_array[rows, :, -1] = frame_array[rows, :, 0].astype(np.float32) * np.float32(1.0 / 255.0)
    return time.perf_counter() - started


def _latest_uint8_frames_from_stack(*, np: Any, stack: Any) -> Any:
    stack_array = np.asarray(stack, dtype=np.float32)
    expected = (
        stack_array.shape[0],
        PLAYER_COUNT,
        POLICY_FRAME_STACK_DEPTH,
        TARGET_SIZE,
        TARGET_SIZE,
    )
    if stack_array.shape != expected:
        raise ValueError(f"stack must have shape {expected}, got {stack_array.shape}")
    latest = np.clip(np.rint(stack_array[:, :, -1] * np.float32(255.0)), 0, 255).astype(np.uint8)
    return latest[:, :, None, :, :]


def _stack_row_indices(*, np: Any, batch_size: int, row_mask: Any | None) -> Any:
    if row_mask is None:
        return np.arange(int(batch_size), dtype=np.int64)
    mask = np.asarray(row_mask, dtype=bool)
    if mask.shape != (int(batch_size),):
        raise ValueError(f"row_mask must have shape ({batch_size},), got {mask.shape}")
    return np.flatnonzero(mask).astype(np.int64)


def _row_major_render_rows(*, np: Any, batch_size: int) -> Any:
    return np.repeat(np.arange(int(batch_size), dtype=np.int64), PLAYER_COUNT)


def _row_major_render_players(*, np: Any, batch_size: int) -> Any:
    return np.tile(np.arange(PLAYER_COUNT, dtype=np.int64), int(batch_size))


def _render_cpu_reference_frames(
    *,
    renderer: CpuOracleBatchedObservationRenderer,
    env: VectorMultiplayerEnv,
    row_indices: Any,
    controlled_players: Any,
    out: Any,
    reference_stacks: Any,
    row_mask: Any | None = None,
    reset_selected_rows: bool = False,
) -> tuple[Any, float]:
    request = SourceStateBatchedRenderRequest(
        state=env.state,
        row_indices=row_indices,
        controlled_players=controlled_players,
        out=out,
    )
    result = renderer.render(request)
    flat_frames = result.frames
    row_major = flat_frames.reshape(
        int(env.batch_size),
        PLAYER_COUNT,
        1,
        TARGET_SIZE,
        TARGET_SIZE,
    ).copy()
    stack_sec = _push_row_major_frames_into_stack(
        reference_stacks,
        row_major,
        row_mask=row_mask,
        reset_selected_rows=reset_selected_rows,
    )
    return row_major, _cpu_reference_render_stack_sec(result.telemetry) + stack_sec


def _candidate_render_total_sec(timings: dict[str, float]) -> float:
    return float(
        timings["production_to_compact_sec"]
        + timings["owner_ordered_pack_sec"]
        + timings["host_to_device_sec"]
        + timings["device_render_sec"]
        + timings["device_to_host_sec"]
        + timings["view_major_to_row_major_sec"]
    )


def _joint_actions_for_profile(
    *,
    np: Any,
    controlled_actions: Any,
    other_actions: Any,
    batch_size: int,
) -> Any:
    controlled = np.asarray(controlled_actions, dtype=np.int16)
    other = np.asarray(other_actions, dtype=np.int16)
    if controlled.shape != (int(batch_size),):
        raise ValueError(
            f"controlled_actions must have shape ({batch_size},), got {controlled.shape}"
        )
    if other.shape != (int(batch_size),):
        raise ValueError(f"other_actions must have shape ({batch_size},), got {other.shape}")
    joint = np.full(
        (int(batch_size), PLAYER_COUNT),
        SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID,
        dtype=np.int16,
    )
    joint[:, 0] = controlled
    joint[:, 1] = other
    return joint


def _cpu_reference_render_stack_sec(telemetry: dict[str, float]) -> float:
    return float(
        telemetry.get("pack_sec", 0.0)
        + telemetry.get("render_sec", 0.0)
        + telemetry.get("readback_sec", 0.0)
        + telemetry.get("stack_sec", 0.0)
    )


def _assert_parity(
    *,
    label: str,
    candidate_frames: Any,
    reference_frames: Any,
    candidate_stacks: Any,
    reference_stacks: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    raw = _compare_for_parity(
        label=f"{label}.raw_frames",
        candidate=candidate_frames,
        reference=reference_frames,
        config=config,
        stack_values=False,
    )
    stack = _compare_for_parity(
        label=f"{label}.stacks",
        candidate=candidate_stacks,
        reference=reference_stacks,
        config=config,
        stack_values=True,
    )
    return {"label": label, "raw_frames": raw, "stacks": stack}


def _assert_final_observation_parity(
    *,
    label: str,
    candidate_final_observation: Any,
    reference_final_observation: Any,
    done_mask: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    mask = np.asarray(done_mask, dtype=bool)
    if not bool(mask.any()):
        return {"label": label, "final_observation": _empty_parity_compare(label)}
    result = _compare_for_parity(
        label=f"{label}.final_observation",
        candidate=candidate_final_observation[mask],
        reference=reference_final_observation[mask],
        config=config,
        stack_values=True,
    )
    return {"label": label, "final_observation": result}


def _compare_for_parity(
    *,
    label: str,
    candidate: Any,
    reference: Any,
    config: dict[str, Any],
    stack_values: bool,
) -> dict[str, Any]:
    import numpy as np

    lhs = np.asarray(candidate)
    rhs = np.asarray(reference)
    if lhs.shape != rhs.shape:
        raise AssertionError(
            f"parity shape failed at {label}: {_first_mismatch(candidate, reference)}"
        )
    parity_mode = str(config["parity_mode"])
    exact = bool(np.array_equal(lhs, rhs))
    diff = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
    mismatch_mask = diff > 0.0
    mismatch_count = int(mismatch_mask.sum())
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    result = {
        "label": label,
        "exact": exact,
        "mismatch_count": mismatch_count,
        "total_values": int(diff.size),
        "mismatch_fraction": float(mismatch_count / diff.size) if diff.size else 0.0,
        "max_abs_diff": max_abs_diff,
        "sample": _first_mismatch(candidate, reference) if mismatch_count else {"mismatch": "none"},
    }
    if mismatch_count:
        result["sample_plane"] = _first_mismatch_plane_summary(mismatch_mask)
    if exact:
        return result
    if parity_mode == BOUNDARY_PARITY_MODE_EXACT:
        raise AssertionError(f"exact parity failed at {label}: {result['sample']}")

    allowed_abs_diff = float(config["parity_max_abs_diff"])
    if stack_values:
        allowed_abs_diff = allowed_abs_diff / 255.0 + 1.0e-7
    allowed_mismatches = _allowed_mismatch_count(
        total_values=int(diff.size),
        mismatch_fraction=float(config["parity_max_mismatch_fraction"]),
    )
    too_large = max_abs_diff > allowed_abs_diff
    too_many = mismatch_count > allowed_mismatches
    result.update(
        {
            "allowed_abs_diff": allowed_abs_diff,
            "allowed_mismatch_count": allowed_mismatches,
            "tolerated": not too_large and not too_many,
        }
    )
    if too_large or too_many:
        raise AssertionError(
            f"tolerant parity failed at {label}: "
            f"mismatch_count={mismatch_count}/{allowed_mismatches}, "
            f"max_abs_diff={max_abs_diff}/{allowed_abs_diff}, "
            f"sample={result['sample']}"
        )
    return result


def _allowed_mismatch_count(*, total_values: int, mismatch_fraction: float) -> int:
    if total_values <= 0:
        return 0
    return max(1, int(math.ceil(float(total_values) * float(mismatch_fraction))))


def _empty_parity_compare(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "exact": True,
        "mismatch_count": 0,
        "total_values": 0,
        "mismatch_fraction": 0.0,
        "max_abs_diff": 0.0,
        "sample": {"mismatch": "none"},
    }


def _summarize_parity(
    parity_summaries: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for summary in parity_summaries:
        for key, value in summary.items():
            if key == "label":
                continue
            if isinstance(value, dict) and "mismatch_count" in value:
                comparisons.append(value)
    raw_exact = all(
        comparison["exact"]
        for comparison in comparisons
        if str(comparison["label"]).endswith(".raw_frames")
    )
    stacks_exact = all(
        comparison["exact"]
        for comparison in comparisons
        if str(comparison["label"]).endswith(".stacks")
        or str(comparison["label"]).endswith(".final_observation")
    )
    max_abs_diff = max((float(c["max_abs_diff"]) for c in comparisons), default=0.0)
    mismatch_count = sum(int(c["mismatch_count"]) for c in comparisons)
    total_values = sum(int(c.get("total_values", 0)) for c in comparisons)
    max_comparison_mismatch_fraction = max(
        (float(c.get("mismatch_fraction", 0.0)) for c in comparisons),
        default=0.0,
    )
    nonexact = [c for c in comparisons if not bool(c["exact"])]
    return {
        "mode": config["parity_mode"],
        "requested_mode": config["requested_parity_mode"],
        "max_abs_diff_allowed": config["parity_max_abs_diff"],
        "max_mismatch_fraction_allowed": config["parity_max_mismatch_fraction"],
        "raw_frames_exact": raw_exact,
        "stacks_exact": stacks_exact,
        "all_exact": raw_exact and stacks_exact and not nonexact,
        "total_mismatch_count": mismatch_count,
        "total_compared_value_count": total_values,
        "overall_mismatch_fraction": (
            float(mismatch_count / total_values) if total_values else 0.0
        ),
        "max_comparison_mismatch_fraction": max_comparison_mismatch_fraction,
        "max_abs_diff_observed": max_abs_diff,
        "nonexact_comparison_count": len(nonexact),
        "samples": [comparison["sample"] for comparison in nonexact[:8]],
    }


def _first_mismatch(candidate: Any, reference: Any) -> dict[str, Any]:
    import numpy as np

    lhs = np.asarray(candidate)
    rhs = np.asarray(reference)
    if lhs.shape != rhs.shape:
        return {"candidate_shape": list(lhs.shape), "reference_shape": list(rhs.shape)}
    diff = lhs != rhs
    if not bool(diff.any()):
        return {"mismatch": "none"}
    index = tuple(int(v) for v in np.argwhere(diff)[0])
    sample: dict[str, Any] = {
        "index": list(index),
        "candidate_value": _json_scalar(lhs[index]),
        "reference_value": _json_scalar(rhs[index]),
        "absolute_difference": float(abs(float(lhs[index]) - float(rhs[index]))),
    }
    if len(index) == 5:
        sample.update(
            {
                "logical_row": index[0],
                "player_view": index[1],
                "channel": index[2],
                "y": index[3],
                "x": index[4],
            }
        )
    return sample


def _first_mismatch_plane_summary(mismatch_mask: Any) -> dict[str, Any]:
    import numpy as np

    mask = np.asarray(mismatch_mask, dtype=bool)
    if not bool(mask.any()):
        return {"mismatch": "none"}
    index = tuple(int(v) for v in np.argwhere(mask)[0])
    if len(index) < 5:
        return {"index": list(index), "rank": int(mask.ndim)}
    row, player, channel = index[:3]
    plane = mask[row, player, channel]
    coords = np.argwhere(plane)
    if coords.size == 0:
        return {"index": list(index), "rank": int(mask.ndim)}
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return {
        "logical_row": int(row),
        "player_view": int(player),
        "channel": int(channel),
        "plane_mismatch_count": int(coords.shape[0]),
        "plane_mismatch_fraction": float(coords.shape[0] / plane.size),
        "bbox_y_min": int(y_min),
        "bbox_y_max": int(y_max),
        "bbox_x_min": int(x_min),
        "bbox_x_max": int(x_max),
    }


def _json_scalar(value: Any) -> int | float:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    return int(value)


def _summarize_timings(timings: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        field: {
            "median_sec": _median(values),
            "p95_sec": _p95(values),
        }
        for field, values in timings.items()
    }


def _summarize_numeric(values_by_field: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        field: {
            "median": _median(values),
            "p95": _p95(values),
        }
        for field, values in values_by_field.items()
    }


def _median_array(*, np: Any, values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.median(array))


def _p95_array(*, np: Any, values: Any) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.percentile(array, 95, method="nearest"))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return ordered[index]


def _positive_int(value: Any, name: str) -> int:
    checked = int(value)
    if checked < 1:
        raise ValueError(f"{name} must be positive, got {checked}")
    return checked


def _nonnegative_int(value: Any, name: str) -> int:
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be non-negative, got {checked}")
    return checked


def _bool_config(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _version_or_missing(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def _known_gaps() -> list[str]:
    return [
        "profile-only Modal sidecar; not wired into trainers, tournaments, eval, checkpoints, or live runs",
        "fixed first scope: player_count=2 profile rows; death_mode is explicit but still profile-only",
        "terminal/death counters can be reported, but natural death/reward trainer semantics are not promoted by this profile alone",
        "host-stack modes read GPU output back before stack update; resident observation modes keep search/sample observations and replay targets device-owned but remain profile-only",
        "persistent GPU framebuffer backend is profile-only and policy-space direct64; it is not browser pixel parity",
    ]


def _package_versions() -> dict[str, str]:
    return {
        "LightZero": _version_or_missing("LightZero"),
        "DI-engine": _version_or_missing("DI-engine"),
        "jax": _version_or_missing("jax"),
        "jaxlib": _version_or_missing("jaxlib"),
        "numpy": _version_or_missing("numpy"),
        "torch": _version_or_missing("torch"),
    }


@app.local_entrypoint()
def main(
    batch_size: int = 64,
    compute: str = COMPUTE_H100,
    steps: int = 8,
    warmup_steps: int = 2,
    seed: int = 20260515,
    trail_slots: int = 256,
    body_capacity: int = 0,
    dynamic_render_trail_slots: bool = True,
    min_render_trail_slots: int = DEFAULT_DYNAMIC_MIN_RENDER_TRAIL_SLOTS,
    verify_steps: int = 2,
    cpu_reference_interval: int = 1,
    geometry_dtype: str = DEFAULT_BOUNDARY_GEOMETRY_DTYPE,
    parity_mode: str = DEFAULT_BOUNDARY_PARITY_MODE,
    parity_max_abs_diff: int = DEFAULT_BOUNDARY_PARITY_MAX_ABS_DIFF,
    parity_max_mismatch_fraction: float = DEFAULT_BOUNDARY_PARITY_MAX_MISMATCH_FRACTION,
    max_ticks: int = 2000,
    death_mode: str = vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
    include_lightzero_payload_profile: bool = False,
    pickle_lightzero_payload: bool = True,
    include_rnd_meter: bool = False,
    surface_facade_canary: bool = False,
    surface_facade_divergence_canary: bool = False,
    profile_env_manager_canary: bool = False,
    hybrid_observation_canary: bool = False,
    compact_hybrid_output: bool = True,
    hybrid_profile_spawn_result: bool = False,
    actor_count: int = 4,
    hybrid_policy_probe_simulations: int = 0,
    hybrid_policy_probe_channels: int = 16,
    hybrid_batched_stack_probe_simulations: int = 0,
    hybrid_batched_stack_probe_channels: int = 16,
    hybrid_batched_stack_probe_device_latest: bool = False,
    hybrid_stack_storage_dtype: str = "float32",
    hybrid_materialize_scalar_timestep: bool = True,
    hybrid_compact_service_replay_proof: bool = False,
    hybrid_compact_rollout_slab_probe: bool = False,
    hybrid_compact_rollout_slab_sample_gate: bool = False,
    hybrid_compact_rollout_slab_sample_gate_batch_size: int = 0,
    hybrid_compact_rollout_slab_sample_gate_interval: int = 1,
    hybrid_compact_rollout_slab_sample_gate_replay_pair_capacity: int = 4096,
    hybrid_compact_rollout_slab_learner_gate: bool = False,
    hybrid_compact_rollout_slab_learner_gate_train_steps: int = 1,
    hybrid_compact_rollout_slab_learner_gate_device: str = "cuda",
    hybrid_compact_rollout_slab_learner_gate_include_rnd: bool = False,
    hybrid_compact_rollout_slab_learner_gate_impl: str = (
        COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_TOY_PROBE
    ),
    hybrid_compact_rollout_slab_learner_gate_support_scale: int = 1,
    hybrid_compact_rollout_slab_learner_gate_num_unroll_steps: int = 1,
    hybrid_compact_rollout_slab_action_mode: str = (
        COMPACT_ROLLOUT_SLAB_ACTION_MODE_SEARCH_FEEDBACK
    ),
    hybrid_compact_owned_loop_entrypoint: bool = False,
    hybrid_compact_owned_loop_policy_version_ref: str = "",
    hybrid_compact_owned_loop_model_version_ref: str = "",
    hybrid_compact_owned_loop_policy_source: str = "",
    hybrid_compact_owned_loop_capture_replay_store_state: bool = False,
    hybrid_compact_root_tape_compare: bool = False,
    hybrid_compact_root_tape_max_records: int = 4,
    hybrid_compact_root_tape_allow_resident_host_snapshot: bool = False,
    hybrid_compact_root_tape_compare_fixed_shape_floor: bool = True,
    hybrid_compact_root_tape_compare_mctx: bool = False,
    hybrid_compact_root_tape_compare_model_compile: bool = False,
    hybrid_compact_root_tape_compare_direct_core: bool = False,
    hybrid_compact_root_tape_model_compile_mode: str = "default",
    hybrid_compact_root_tape_require_model_compile: bool = True,
    hybrid_compact_root_tape_reference_label: str = "primary",
    hybrid_device_only_stack: bool = False,
    hybrid_refresh_observation_stack: bool = True,
    hybrid_resident_observation_search: bool = False,
    hybrid_native_actor_buffer: bool = False,
    hybrid_persistent_compact_render_state_buffer: bool = False,
    hybrid_borrow_single_actor_render_state: bool = False,
    hybrid_resident_chunk_probe: bool = False,
    hybrid_resident_replay_steps: int = 64,
    hybrid_resident_sample_batch_size: int = 256,
    hybrid_resident_replay_train_steps: int = 1,
    hybrid_resident_readback_checksum: bool = True,
    hybrid_lightzero_collect_forward_probe: bool = False,
    hybrid_lightzero_initial_inference_probe: bool = False,
    hybrid_lightzero_array_ceiling_probe: bool = False,
    hybrid_lightzero_mcts_arrays_boundary_probe: bool = False,
    hybrid_mctx_compact_search_probe: bool = False,
    hybrid_lightzero_mcts_arrays_boundary_impl: str = (
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_STOCK_FACADE
    ),
    hybrid_lightzero_mcts_arrays_boundary_input_mode: str = (
        LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8
    ),
    hybrid_lightzero_array_ceiling_mode: str = LIGHTZERO_ARRAY_CEILING_MODE_POLICY_ARRAYS,
    hybrid_lightzero_array_ceiling_input_mode: str = (
        LIGHTZERO_ARRAY_CEILING_INPUT_MODE_HOST_UINT8
    ),
    hybrid_lightzero_mock_service_materialize_public_output: bool = False,
    hybrid_lightzero_consumer_num_simulations: int = 8,
    hybrid_lightzero_consumer_temperature: float = 1.0,
    hybrid_lightzero_consumer_epsilon: float = 0.0,
    hybrid_lightzero_consumer_root_noise_weight: float = -1.0,
    hybrid_lightzero_consumer_use_cuda: bool = True,
    hybrid_lightzero_consumer_collect_with_pure_policy: bool = False,
    hybrid_compact_torch_compile_search: bool = True,
    hybrid_compact_torch_compile_model_inference: bool = False,
    hybrid_compact_torch_require_model_compile: bool = False,
    hybrid_compact_torch_model_compile_mode: str = "reduce-overhead",
    hybrid_compact_torch_recurrent_action_shape_mode: str = "auto",
    hybrid_compact_torch_timing_mode: str = "host_phase_sync",
    hybrid_compact_torch_initial_inference_mode: str = "model_method",
    hybrid_compact_torch_observation_memory_format: str = "contiguous",
    hybrid_compact_torch_model_memory_format: str = "contiguous",
    hybrid_mctx_num_simulations: int = 8,
    hybrid_mctx_hidden_dim: int = 64,
    hybrid_mctx_visual_channels: int = 8,
    hybrid_mctx_require_gpu_backend: bool = True,
    hybrid_mctx_lightzero_checkpoint_ref: str = "",
    hybrid_mctx_lightzero_checkpoint_state_key: str = "",
    hybrid_mctx_compare_direct_ctree: bool = False,
    hybrid_mctx_compare_direct_ctree_impl: str = (
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT
    ),
    surface_stack_backend: str = TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
    observation_renderer_backend: str = SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND,
    render_surface: str = RENDER_SURFACE_BLOCK_704_GRAY64,
    allow_render_truncation: bool = False,
    prewarm_render_functions: bool = False,
    rnd_batch_size: int = 64,
    rnd_update_per_collect: int = xb.RND_DEFAULT_UPDATE_PER_COLLECT,
    rnd_device: str = "cpu",
) -> None:
    if compute not in COMPUTE_CHOICES:
        allowed = ", ".join(sorted(COMPUTE_CHOICES))
        raise ValueError(f"compute must be one of {allowed}; got {compute!r}")
    config = {
        "batch_size": batch_size,
        "compute": compute,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "trail_slots": trail_slots,
        "body_capacity": body_capacity if int(body_capacity) > 0 else trail_slots,
        "dynamic_render_trail_slots": dynamic_render_trail_slots,
        "min_render_trail_slots": min_render_trail_slots,
        "verify_steps": verify_steps,
        "cpu_reference_interval": cpu_reference_interval,
        "geometry_dtype": geometry_dtype,
        "parity_mode": parity_mode,
        "parity_max_abs_diff": parity_max_abs_diff,
        "parity_max_mismatch_fraction": parity_max_mismatch_fraction,
        "max_ticks": max_ticks,
        "death_mode": death_mode,
        "include_lightzero_payload_profile": include_lightzero_payload_profile,
        "pickle_lightzero_payload": pickle_lightzero_payload,
        "include_rnd_meter": include_rnd_meter,
        "surface_facade_canary": surface_facade_canary,
        "surface_facade_divergence_canary": surface_facade_divergence_canary,
        "profile_env_manager_canary": profile_env_manager_canary,
        "hybrid_observation_canary": hybrid_observation_canary,
        "compact_hybrid_output": compact_hybrid_output,
        "actor_count": actor_count,
        "hybrid_policy_probe_simulations": hybrid_policy_probe_simulations,
        "hybrid_policy_probe_channels": hybrid_policy_probe_channels,
        "hybrid_batched_stack_probe_simulations": hybrid_batched_stack_probe_simulations,
        "hybrid_batched_stack_probe_channels": hybrid_batched_stack_probe_channels,
        "hybrid_batched_stack_probe_device_latest": hybrid_batched_stack_probe_device_latest,
        "hybrid_stack_storage_dtype": hybrid_stack_storage_dtype,
        "hybrid_materialize_scalar_timestep": hybrid_materialize_scalar_timestep,
        "hybrid_compact_service_replay_proof": hybrid_compact_service_replay_proof,
        "hybrid_compact_rollout_slab_probe": hybrid_compact_rollout_slab_probe,
        "hybrid_compact_rollout_slab_sample_gate": hybrid_compact_rollout_slab_sample_gate,
        "hybrid_compact_rollout_slab_sample_gate_batch_size": (
            hybrid_compact_rollout_slab_sample_gate_batch_size
        ),
        "hybrid_compact_rollout_slab_sample_gate_interval": (
            hybrid_compact_rollout_slab_sample_gate_interval
        ),
        "hybrid_compact_rollout_slab_sample_gate_replay_pair_capacity": (
            hybrid_compact_rollout_slab_sample_gate_replay_pair_capacity
        ),
        "hybrid_compact_rollout_slab_learner_gate": hybrid_compact_rollout_slab_learner_gate,
        "hybrid_compact_rollout_slab_learner_gate_train_steps": (
            hybrid_compact_rollout_slab_learner_gate_train_steps
        ),
        "hybrid_compact_rollout_slab_learner_gate_device": (
            hybrid_compact_rollout_slab_learner_gate_device
        ),
        "hybrid_compact_rollout_slab_learner_gate_include_rnd": (
            hybrid_compact_rollout_slab_learner_gate_include_rnd
        ),
        "hybrid_compact_rollout_slab_learner_gate_impl": (
            hybrid_compact_rollout_slab_learner_gate_impl
        ),
        "hybrid_compact_rollout_slab_learner_gate_support_scale": (
            hybrid_compact_rollout_slab_learner_gate_support_scale
        ),
        "hybrid_compact_rollout_slab_learner_gate_num_unroll_steps": (
            hybrid_compact_rollout_slab_learner_gate_num_unroll_steps
        ),
        "hybrid_compact_rollout_slab_action_mode": hybrid_compact_rollout_slab_action_mode,
        "hybrid_compact_owned_loop_entrypoint": hybrid_compact_owned_loop_entrypoint,
        "hybrid_compact_owned_loop_policy_version_ref": (
            hybrid_compact_owned_loop_policy_version_ref
        ),
        "hybrid_compact_owned_loop_model_version_ref": (
            hybrid_compact_owned_loop_model_version_ref
        ),
        "hybrid_compact_owned_loop_policy_source": (hybrid_compact_owned_loop_policy_source),
        "hybrid_compact_owned_loop_capture_replay_store_state": (
            hybrid_compact_owned_loop_capture_replay_store_state
        ),
        "hybrid_compact_root_tape_compare": hybrid_compact_root_tape_compare,
        "hybrid_compact_root_tape_max_records": hybrid_compact_root_tape_max_records,
        "hybrid_compact_root_tape_allow_resident_host_snapshot": (
            hybrid_compact_root_tape_allow_resident_host_snapshot
        ),
        "hybrid_compact_root_tape_compare_fixed_shape_floor": (
            hybrid_compact_root_tape_compare_fixed_shape_floor
        ),
        "hybrid_compact_root_tape_compare_mctx": hybrid_compact_root_tape_compare_mctx,
        "hybrid_compact_root_tape_compare_model_compile": (
            hybrid_compact_root_tape_compare_model_compile
        ),
        "hybrid_compact_root_tape_compare_direct_core": (
            hybrid_compact_root_tape_compare_direct_core
        ),
        "hybrid_compact_root_tape_model_compile_mode": (
            hybrid_compact_root_tape_model_compile_mode
        ),
        "hybrid_compact_root_tape_require_model_compile": (
            hybrid_compact_root_tape_require_model_compile
        ),
        "hybrid_compact_root_tape_reference_label": (hybrid_compact_root_tape_reference_label),
        "hybrid_device_only_stack": hybrid_device_only_stack,
        "hybrid_refresh_observation_stack": hybrid_refresh_observation_stack,
        "hybrid_resident_observation_search": hybrid_resident_observation_search,
        "hybrid_native_actor_buffer": hybrid_native_actor_buffer,
        "hybrid_persistent_compact_render_state_buffer": (
            hybrid_persistent_compact_render_state_buffer
        ),
        "hybrid_borrow_single_actor_render_state": hybrid_borrow_single_actor_render_state,
        "hybrid_resident_chunk_probe": hybrid_resident_chunk_probe,
        "hybrid_resident_replay_steps": hybrid_resident_replay_steps,
        "hybrid_resident_sample_batch_size": hybrid_resident_sample_batch_size,
        "hybrid_resident_replay_train_steps": hybrid_resident_replay_train_steps,
        "hybrid_resident_readback_checksum": hybrid_resident_readback_checksum,
        "hybrid_lightzero_collect_forward_probe": hybrid_lightzero_collect_forward_probe,
        "hybrid_lightzero_initial_inference_probe": hybrid_lightzero_initial_inference_probe,
        "hybrid_lightzero_array_ceiling_probe": hybrid_lightzero_array_ceiling_probe,
        "hybrid_lightzero_mcts_arrays_boundary_probe": (
            hybrid_lightzero_mcts_arrays_boundary_probe
        ),
        "hybrid_mctx_compact_search_probe": hybrid_mctx_compact_search_probe,
        "hybrid_lightzero_mcts_arrays_boundary_impl": (hybrid_lightzero_mcts_arrays_boundary_impl),
        "hybrid_lightzero_mcts_arrays_boundary_input_mode": (
            hybrid_lightzero_mcts_arrays_boundary_input_mode
        ),
        "hybrid_lightzero_array_ceiling_mode": hybrid_lightzero_array_ceiling_mode,
        "hybrid_lightzero_array_ceiling_input_mode": (hybrid_lightzero_array_ceiling_input_mode),
        "hybrid_lightzero_mock_service_materialize_public_output": (
            hybrid_lightzero_mock_service_materialize_public_output
        ),
        "hybrid_lightzero_consumer_num_simulations": hybrid_lightzero_consumer_num_simulations,
        "hybrid_lightzero_consumer_temperature": hybrid_lightzero_consumer_temperature,
        "hybrid_lightzero_consumer_epsilon": hybrid_lightzero_consumer_epsilon,
        "hybrid_lightzero_consumer_root_noise_weight": (
            hybrid_lightzero_consumer_root_noise_weight
        ),
        "hybrid_lightzero_consumer_use_cuda": hybrid_lightzero_consumer_use_cuda,
        "hybrid_lightzero_consumer_collect_with_pure_policy": (
            hybrid_lightzero_consumer_collect_with_pure_policy
        ),
        "hybrid_compact_torch_compile_search": hybrid_compact_torch_compile_search,
        "hybrid_compact_torch_compile_model_inference": (
            hybrid_compact_torch_compile_model_inference
        ),
        "hybrid_compact_torch_require_model_compile": hybrid_compact_torch_require_model_compile,
        "hybrid_compact_torch_model_compile_mode": (hybrid_compact_torch_model_compile_mode),
        "hybrid_compact_torch_recurrent_action_shape_mode": (
            hybrid_compact_torch_recurrent_action_shape_mode
        ),
        "hybrid_compact_torch_timing_mode": hybrid_compact_torch_timing_mode,
        "hybrid_compact_torch_initial_inference_mode": (
            hybrid_compact_torch_initial_inference_mode
        ),
        "hybrid_compact_torch_observation_memory_format": (
            hybrid_compact_torch_observation_memory_format
        ),
        "hybrid_compact_torch_model_memory_format": (hybrid_compact_torch_model_memory_format),
        "hybrid_mctx_num_simulations": hybrid_mctx_num_simulations,
        "hybrid_mctx_hidden_dim": hybrid_mctx_hidden_dim,
        "hybrid_mctx_visual_channels": hybrid_mctx_visual_channels,
        "hybrid_mctx_require_gpu_backend": hybrid_mctx_require_gpu_backend,
        "hybrid_mctx_lightzero_checkpoint_ref": hybrid_mctx_lightzero_checkpoint_ref,
        "hybrid_mctx_lightzero_checkpoint_state_key": hybrid_mctx_lightzero_checkpoint_state_key,
        "hybrid_mctx_compare_direct_ctree": hybrid_mctx_compare_direct_ctree,
        "hybrid_mctx_compare_direct_ctree_impl": hybrid_mctx_compare_direct_ctree_impl,
        "surface_stack_backend": surface_stack_backend,
        "observation_renderer_backend": observation_renderer_backend,
        "render_surface": render_surface,
        "allow_render_truncation": allow_render_truncation,
        "prewarm_render_functions": prewarm_render_functions,
        "rnd_batch_size": rnd_batch_size,
        "rnd_update_per_collect": rnd_update_per_collect,
        "rnd_device": rnd_device,
    }
    fn = _select_profile_function(
        compute=compute,
        hybrid_observation_canary=hybrid_observation_canary,
        hybrid_mctx_compact_search_probe=hybrid_mctx_compact_search_probe,
        hybrid_compact_root_tape_compare_mctx=hybrid_compact_root_tape_compare_mctx,
        hybrid_compact_root_tape_compare_model_compile=(
            hybrid_compact_root_tape_compare_model_compile
        ),
        hybrid_compact_root_tape_compare_direct_core=(hybrid_compact_root_tape_compare_direct_core),
        hybrid_mctx_lightzero_checkpoint_ref=hybrid_mctx_lightzero_checkpoint_ref,
        hybrid_lightzero_collect_forward_probe=hybrid_lightzero_collect_forward_probe,
        hybrid_lightzero_initial_inference_probe=hybrid_lightzero_initial_inference_probe,
        hybrid_lightzero_array_ceiling_probe=hybrid_lightzero_array_ceiling_probe,
        hybrid_lightzero_mcts_arrays_boundary_probe=hybrid_lightzero_mcts_arrays_boundary_probe,
        profile_env_manager_canary=profile_env_manager_canary,
        surface_facade_canary=surface_facade_canary,
        include_rnd_meter=include_rnd_meter,
    )
    if hybrid_profile_spawn_result:
        call = fn.spawn(config)
        launch = {
            "schema_id": "curvyzero_hybrid_observation_profile_spawn/v0",
            "status": "spawned",
            "launch_mode": "detached_function_call_result",
            "result_capture": "modal_function_call_get",
            "function_call_id": str(call.object_id),
            "profile_only": True,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "app_name": APP_NAME,
            "compute": compute,
            "death_mode": death_mode,
            "hybrid_observation_canary": hybrid_observation_canary,
            "compact_owned_loop_entrypoint": hybrid_compact_owned_loop_entrypoint,
            "compact_owned_loop_policy_version_ref": (hybrid_compact_owned_loop_policy_version_ref),
            "compact_root_tape_compare": hybrid_compact_root_tape_compare,
            "compact_root_tape_max_records": hybrid_compact_root_tape_max_records,
            "hybrid_borrow_single_actor_render_state": (
                hybrid_borrow_single_actor_render_state
            ),
        }
        print(json.dumps(launch, indent=2, sort_keys=True))
        return
    result = fn.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not bool(result.get("ok", False)):
        raise RuntimeError(result.get("error", "boundary profile failed"))


__all__ = [
    "APP_NAME",
    "BOUNDARY_PARITY_MODE_EXACT",
    "BOUNDARY_PARITY_MODE_TOLERANT",
    "COMPUTE_H100",
    "COMPUTE_H100_CPU64",
    "COMPUTE_L4_T4",
    "DEFAULT_BOUNDARY_PARITY_MAX_ABS_DIFF",
    "DEFAULT_BOUNDARY_PARITY_MAX_MISMATCH_FRACTION",
    "DEFAULT_BOUNDARY_PARITY_MODE",
    "DEFAULT_BOUNDARY_GEOMETRY_DTYPE",
    "SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND",
    "SCHEMA_ID",
    "_assert_parity",
    "_push_row_major_frames_into_stack",
    "_row_major_render_players",
    "_row_major_render_rows",
    "_assert_no_render_truncation_if_required",
    "_select_render_trail_slots",
    "_trail_stats_after_owner_pack",
    "_truncate_compact_trails_for_render",
    "_view_major_to_row_major_frames",
]
