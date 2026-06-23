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
      --observation-mode curvytron_visual_root \
      --compute h100 \
      --batch-size 512 \
      --player-count 2 \
      --num-simulations 16 \
      --hidden-dim 64 \
      --max-depth 16 \
      --warmup-runs 2 \
      --steady-runs 5 \
      --legal-mask-profile mixed_2of3

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

    uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
      --observation-mode curvytron_hybrid_compact_visual_sample \
      --compute h100 \
      --batch-size 64 \
      --player-count 2 \
      --body-capacity 4096 \
      --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
      --rollout-steps 4 \
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
legal policy rows, and then runs the same synthetic Mctx search timing. The
`curvytron_hybrid_compact_visual_sample` mode builds a renderer-backed
`HybridCompactBatch` with real `[B,2,4,64,64]` stacks, validates
`CompactRootBatchV1`, then runs the same Mctx timing and validates
`CompactSearchResultV1`. In all CurvyTron-shaped modes, `--batch-size` is the
env-row count.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import functools
import json
from pathlib import Path
import statistics
import subprocess
import time
from importlib import metadata
from typing import Any

import modal
import numpy as np

from curvyzero.infra.modal.mctx_dependency_smoke import JAX_VERSION
from curvyzero.infra.modal.mctx_dependency_smoke import MCTX_VERSION

APP_NAME = "curvyzero-mctx-synthetic-benchmark"
REMOTE_ROOT = Path("/repo")
ACTION_COUNT = 3
MCTX_COMPACT_VISUAL_SEARCH_SERVICE_BACKEND_NAME = (
    "mctx_hybrid_compact_visual_search_service"
)
MCTX_COMPACT_VISUAL_SEARCH_SERVICE_SEMANTICS = (
    "mctx_gumbel_muzero_policy_synthetic_visual_profile_not_lightzero_ctree"
)
OBSERVATION_MODE_FLAT = "flat_hidden"
OBSERVATION_MODE_CURVYTRON_VISUAL_ROOT = "curvytron_visual_root"
OBSERVATION_MODE_CURVYTRON_DEBUG = "curvytron_debug"
OBSERVATION_MODE_CURVYTRON_DEBUG_PACKER = "curvytron_debug_packer"
OBSERVATION_MODE_CURVYTRON_ACTOR_BRIDGE_SAMPLE = "curvytron_actor_bridge_sample"
OBSERVATION_MODE_CURVYTRON_TRAINER_FLAT = "curvytron_trainer_flat"
OBSERVATION_MODE_CURVYTRON_VECTOR_TRAINER_SAMPLE = "curvytron_vector_trainer_sample"
OBSERVATION_MODE_CURVYTRON_HYBRID_COMPACT_VISUAL_SAMPLE = (
    "curvytron_hybrid_compact_visual_sample"
)
COMPACT_VISUAL_OBSERVATION_SOURCE_HOST = "host"
COMPACT_VISUAL_OBSERVATION_SOURCE_RESIDENT_GPU = "resident_gpu"
COMPACT_VISUAL_OBSERVATION_SOURCES = {
    COMPACT_VISUAL_OBSERVATION_SOURCE_HOST,
    COMPACT_VISUAL_OBSERVATION_SOURCE_RESIDENT_GPU,
}
COMPACT_VISUAL_EXPECTED_STACK_DTYPE = np.uint8
OBSERVATION_MODES = {
    OBSERVATION_MODE_FLAT,
    OBSERVATION_MODE_CURVYTRON_VISUAL_ROOT,
    OBSERVATION_MODE_CURVYTRON_DEBUG,
    OBSERVATION_MODE_CURVYTRON_DEBUG_PACKER,
    OBSERVATION_MODE_CURVYTRON_ACTOR_BRIDGE_SAMPLE,
    OBSERVATION_MODE_CURVYTRON_TRAINER_FLAT,
    OBSERVATION_MODE_CURVYTRON_VECTOR_TRAINER_SAMPLE,
    OBSERVATION_MODE_CURVYTRON_HYBRID_COMPACT_VISUAL_SAMPLE,
}
CURVYTRON_SHAPED_OBSERVATION_MODES = {
    OBSERVATION_MODE_CURVYTRON_VISUAL_ROOT,
    OBSERVATION_MODE_CURVYTRON_DEBUG,
    OBSERVATION_MODE_CURVYTRON_DEBUG_PACKER,
    OBSERVATION_MODE_CURVYTRON_ACTOR_BRIDGE_SAMPLE,
    OBSERVATION_MODE_CURVYTRON_TRAINER_FLAT,
    OBSERVATION_MODE_CURVYTRON_VECTOR_TRAINER_SAMPLE,
    OBSERVATION_MODE_CURVYTRON_HYBRID_COMPACT_VISUAL_SAMPLE,
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


def _validate_resident_compact_visual_latest(
    latest: Any,
    *,
    env_rows: int,
    players: int,
) -> None:
    latest_shape = tuple(int(dim) for dim in getattr(latest, "shape", ()))
    expected_shape = (int(env_rows), int(players), 1, 64, 64)
    if latest_shape != expected_shape:
        raise ValueError(
            "resident compact visual latest-frame shape mismatch; "
            f"got {latest_shape}, expected {expected_shape}"
        )
    latest_dtype = getattr(latest, "dtype", None)
    if latest_dtype is None or np.dtype(latest_dtype) != COMPACT_VISUAL_EXPECTED_STACK_DTYPE:
        raise ValueError(
            "resident compact visual latest-frame dtype must be uint8, "
            f"got {latest_dtype}"
        )


def _validate_compact_visual_root_row_major_order(
    *,
    env_rows: Any,
    players: Any,
    env_row_count: int,
    player_count: int,
) -> None:
    root_rows = np.asarray(env_rows, dtype=np.int64).reshape(-1)
    root_players = np.asarray(players, dtype=np.int64).reshape(-1)
    expected_rows = np.repeat(np.arange(int(env_row_count), dtype=np.int64), int(player_count))
    expected_players = np.tile(
        np.arange(int(player_count), dtype=np.int64),
        int(env_row_count),
    )
    if root_rows.shape != expected_rows.shape or root_players.shape != expected_players.shape:
        raise ValueError(
            "compact visual root row/player shape mismatch; "
            f"got rows={root_rows.shape}, players={root_players.shape}, "
            f"expected={expected_rows.shape}"
        )
    if not np.array_equal(root_rows, expected_rows) or not np.array_equal(
        root_players,
        expected_players,
    ):
        raise ValueError(
            "resident compact visual mode requires row-major [env row, player] root order"
        )


def _compact_benchmark_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Small run summary for Modal profile grids."""

    timing = result.get("timing", {})
    output = result.get("output", {})
    contract = output.get("closed_compact_loop_contract") or {}
    bucket_totals = timing.get("closed_compact_loop_bucket_totals_sec") or {}
    next_step_totals = timing.get("closed_compact_loop_next_step_bucket_totals_sec") or {}
    bucket_fractions = timing.get("closed_compact_loop_bucket_fraction_of_total") or {}
    config = result.get("config", {})
    return {
        "ok": result.get("ok"),
        "problems": result.get("problems"),
        "run_shape": {
            "compute": config.get("compute"),
            "batch_size": config.get("batch_size"),
            "root_batch_size": config.get("root_batch_size"),
            "num_simulations": config.get("num_simulations"),
            "closed_loop_steps": config.get("closed_loop_steps"),
            "closed_loop_action_only_profile": config.get(
                "closed_loop_action_only_profile"
            ),
            "closed_loop_deferred_payload_profile": config.get(
                "closed_loop_deferred_payload_profile"
            ),
            "closed_loop_overlap_payload_profile": config.get(
                "closed_loop_overlap_payload_profile"
            ),
            "compact_visual_observation_source": config.get(
                "compact_visual_observation_source"
            ),
            "closed_loop_replay_index": config.get("closed_loop_replay_index"),
            "native_actor_buffer": config.get("native_actor_buffer"),
            "hybrid_refresh_observation_stack": config.get(
                "hybrid_refresh_observation_stack"
            ),
            "hybrid_borrow_single_actor_render_state": config.get(
                "hybrid_borrow_single_actor_render_state"
            ),
            "hybrid_persistent_compact_render_state_buffer": config.get(
                "hybrid_persistent_compact_render_state_buffer"
            ),
            "compact_visual_resident_sync": config.get(
                "compact_visual_resident_sync"
            ),
            "persistent_renderer_async_device_only_profile": config.get(
                "persistent_renderer_async_device_only_profile"
            ),
            "persistent_vectorized_delta_pack_profile": config.get(
                "persistent_vectorized_delta_pack_profile"
            ),
            "compact_root_copy_observation": config.get(
                "compact_root_copy_observation"
            ),
        },
        "closed_loop": {
            "active_roots_per_sec": timing.get(
                "closed_compact_loop_active_roots_per_sec"
            ),
            "slowest_bucket": timing.get("closed_compact_loop_slowest_bucket"),
            "completed_steps": contract.get("completed_steps"),
            "total_active_roots": contract.get("total_active_roots"),
            "total_sec": contract.get("total_sec"),
            "action_loop_total_sec": contract.get("action_loop_total_sec"),
            "deferred_search_payload_flush_sec": contract.get(
                "deferred_search_payload_flush_sec"
            ),
            "deferred_search_payload_bytes": contract.get(
                "deferred_search_payload_bytes"
            ),
            "deferred_search_payload_count": contract.get(
                "deferred_search_payload_count"
            ),
            "overlapped_search_payload_wait_sec": contract.get(
                "overlapped_search_payload_wait_sec"
            ),
            "overlapped_search_payload_bytes": contract.get(
                "overlapped_search_payload_bytes"
            ),
            "overlapped_search_payload_count": contract.get(
                "overlapped_search_payload_count"
            ),
            "bucket_totals_sec": bucket_totals,
            "next_step_bucket_totals_sec": next_step_totals,
            "bucket_fraction_of_total": bucket_fractions,
            "plain_breakdown": _closed_compact_plain_breakdown(
                bucket_totals=bucket_totals,
                next_step_totals=next_step_totals,
                total_sec=contract.get("total_sec"),
            ),
        },
        "one_step": {
            "host_observation_setup_sec": timing.get("host_observation_setup_sec"),
            "host_setup_plus_fresh_boundary_sec": timing.get(
                "host_setup_plus_fresh_boundary_sec"
            ),
            "host_setup_plus_fresh_boundary_active_decisions_per_sec": timing.get(
                "host_setup_plus_fresh_boundary_active_decisions_per_sec"
            ),
            "end_to_end_active_decisions_per_sec_median": timing.get(
                "end_to_end_active_decisions_per_sec_median"
            ),
            "end_to_end_h2d_median_sec": timing.get("end_to_end_h2d_median_sec"),
            "end_to_end_search_median_sec": timing.get(
                "end_to_end_search_median_sec"
            ),
            "end_to_end_d2h_median_sec": timing.get("end_to_end_d2h_median_sec"),
        },
        "compact_search_service_profile": result.get("compact_search_service_profile"),
    }


def _mctx_compact_visual_search_service_profile_row(
    *,
    observation_mode: str,
    num_simulations: int,
    end_to_end_active_decisions_per_sec_median: float | None,
    closed_compact_loop_active_roots_per_sec: float | None,
    compact_search_contract: dict[str, Any] | None,
    compact_replay_index_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    """Comparable MCTX profile row without claiming stock LightZero semantics."""

    return {
        "schema_id": "curvyzero_compact_search_service_profile/v1",
        "backend_name": MCTX_COMPACT_VISUAL_SEARCH_SERVICE_BACKEND_NAME,
        "semantics": MCTX_COMPACT_VISUAL_SEARCH_SERVICE_SEMANTICS,
        "profile_only": True,
        "not_lightzero_ctree": True,
        "not_train_muzero": True,
        "observation_mode": str(observation_mode),
        "num_simulations": int(num_simulations),
        "metrics": {
            "end_to_end_active_decisions_per_sec_median": (
                None
                if end_to_end_active_decisions_per_sec_median is None
                else float(end_to_end_active_decisions_per_sec_median)
            ),
            "closed_compact_loop_active_roots_per_sec": (
                None
                if closed_compact_loop_active_roots_per_sec is None
                else float(closed_compact_loop_active_roots_per_sec)
            ),
        },
        "contracts": {
            "compact_search_contract_present": compact_search_contract is not None,
            "compact_replay_index_contract_present": (
                compact_replay_index_contract is not None
            ),
        },
    }


def _closed_compact_plain_breakdown(
    *,
    bucket_totals: dict[str, Any],
    next_step_totals: dict[str, Any],
    total_sec: Any,
) -> dict[str, Any]:
    """Group closed-loop timings so env mechanics are not confused with obs handoff."""

    def seconds(mapping: dict[str, Any], key: str) -> float:
        try:
            return float(mapping.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    total = 0.0
    try:
        total = float(total_sec or 0.0)
    except (TypeError, ValueError):
        total = 0.0

    top_level = {
        "env_step_sec": seconds(bucket_totals, "env_step_sec"),
        "search_sec": seconds(bucket_totals, "search_sec"),
        "root_build_sec": seconds(bucket_totals, "root_build_sec"),
        "root_sidecar_sec": seconds(bucket_totals, "root_sidecar_sec"),
        "h2d_sec": seconds(bucket_totals, "h2d_sec"),
        "d2h_sec": seconds(bucket_totals, "d2h_sec"),
        "root_value_extract_sec": seconds(bucket_totals, "root_value_extract_sec"),
        "search_result_validate_sec": seconds(
            bucket_totals,
            "search_result_validate_sec",
        ),
        "deferred_search_payload_flush_sec": seconds(
            bucket_totals,
            "deferred_search_payload_flush_sec",
        ),
        "overlapped_search_payload_wait_sec": seconds(
            bucket_totals,
            "overlapped_search_payload_wait_sec",
        ),
        "joint_action_build_sec": seconds(bucket_totals, "joint_action_build_sec"),
        "replay_index_sec": seconds(bucket_totals, "replay_index_sec"),
    }
    top_level_sum = sum(top_level.values())
    top_level["unlabeled_residual_sec"] = max(0.0, total - top_level_sum)

    game_mechanics = (
        seconds(next_step_totals, "actor_env_runtime_sec")
        + seconds(next_step_totals, "actor_env_reward_sec")
        + seconds(next_step_totals, "actor_env_post_runtime_bookkeeping_sec")
    )
    public_packaging = (
        seconds(next_step_totals, "actor_env_public_prepare_sec")
        + seconds(next_step_totals, "actor_env_public_info_sec")
        + seconds(next_step_totals, "actor_env_batch_pack_sec")
        + seconds(next_step_totals, "actor_compact_write_sec")
    )
    observation_handoff = (
        seconds(next_step_totals, "actor_render_state_write_sec")
        + seconds(next_step_totals, "observation_sec")
        + seconds(next_step_totals, "resident_stack_update_sec")
    )
    host_stack_update = (
        seconds(next_step_totals, "stack_shift_sec")
        + seconds(next_step_totals, "stack_latest_update_sec")
    )
    renderer_leaves = {
        "production_to_compact_sec": seconds(
            next_step_totals,
            "renderer_production_to_compact_sec",
        ),
        "persistent_delta_pack_sec": seconds(
            next_step_totals,
            "renderer_persistent_delta_pack_sec",
        ),
        "renderer_h2d_sec": seconds(next_step_totals, "renderer_host_to_device_sec"),
        "persistent_update_sec": seconds(
            next_step_totals,
            "renderer_persistent_update_sec",
        ),
        "gpu_draw_sec": seconds(next_step_totals, "renderer_device_render_sec"),
        "renderer_d2h_sec": seconds(next_step_totals, "renderer_device_to_host_sec"),
    }
    env_step = top_level["env_step_sec"]

    return {
        "note": (
            "top_level_sec is wall time. env_step_leaf_sec is diagnostic "
            "attribution inside env_step_sec; some leaves are nested and should "
            "not be summed as an exclusive profile."
        ),
        "top_level_sec": top_level,
        "top_level_fraction": {
            key: (value / total if total > 0.0 else 0.0)
            for key, value in top_level.items()
        },
        "env_step_leaf_sec": {
            "game_mechanics_leaf_sec": game_mechanics,
            "public_packaging_leaf_sec": public_packaging,
            "observation_handoff_leaf_sec": observation_handoff,
            "actor_render_state_write_sec": seconds(
                next_step_totals,
                "actor_render_state_write_sec",
            ),
            "actor_render_state_write_visual_trail_sec": seconds(
                next_step_totals,
                "actor_render_state_write_visual_trail_sec",
            ),
            "actor_render_state_write_player_sec": seconds(
                next_step_totals,
                "actor_render_state_write_player_sec",
            ),
            "actor_render_state_write_bonus_sec": seconds(
                next_step_totals,
                "actor_render_state_write_bonus_sec",
            ),
            "actor_render_state_write_other_sec": seconds(
                next_step_totals,
                "actor_render_state_write_other_sec",
            ),
            "observation_sec": seconds(next_step_totals, "observation_sec"),
            "resident_stack_update_sec": seconds(
                next_step_totals,
                "resident_stack_update_sec",
            ),
            "host_stack_update_leaf_sec": host_stack_update,
            "renderer_leaf_sec": renderer_leaves,
        },
        "env_step_leaf_fraction_of_env_step": {
            "game_mechanics": (game_mechanics / env_step if env_step > 0.0 else 0.0),
            "public_packaging": (
                public_packaging / env_step if env_step > 0.0 else 0.0
            ),
            "observation_handoff": (
                observation_handoff / env_step if env_step > 0.0 else 0.0
            ),
            "host_stack_update": (
                host_stack_update / env_step if env_step > 0.0 else 0.0
            ),
        },
    }


def _legal_mask_profile(config: dict[str, Any]) -> str:
    value = str(config.get("legal_mask_profile") or "all3")
    allowed = {"all3", "mixed_2of3"}
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"legal_mask_profile must be one of {allowed_text}, got {value!r}")
    return value


def _mctx_legality_summary(
    actions: Any,
    action_weights: Any,
    invalid_actions: Any,
    active_root_mask: Any,
    *,
    illegal_weight_atol: float = 1e-7,
) -> tuple[dict[str, Any], list[str]]:
    """Summarize MCTX output legality against an explicit active-root mask."""

    actions_np = np.asarray(actions)
    weights_np = np.asarray(action_weights, dtype=np.float64)
    invalid_np = np.asarray(invalid_actions, dtype=np.bool_)
    active_np = np.asarray(active_root_mask, dtype=np.bool_)
    problems: list[str] = []

    if invalid_np.ndim != 2:
        raise ValueError(f"invalid_actions must have shape [R,A], got {invalid_np.shape}")
    root_count, action_count = invalid_np.shape
    if action_count != ACTION_COUNT:
        raise ValueError(
            f"invalid_actions second dimension must be {ACTION_COUNT}, got {action_count}"
        )
    if actions_np.shape != (root_count,):
        raise ValueError(f"actions must have shape {(root_count,)}, got {actions_np.shape}")
    if weights_np.shape != (root_count, ACTION_COUNT):
        raise ValueError(
            "action_weights must have shape "
            f"{(root_count, ACTION_COUNT)}, got {weights_np.shape}"
        )
    if active_np.shape != (root_count,):
        raise ValueError(
            f"active_root_mask must have shape {(root_count,)}, got {active_np.shape}"
        )

    active_count = int(active_np.sum())
    active_actions = actions_np[active_np]
    active_invalid = invalid_np[active_np]
    active_weights = weights_np[active_np]
    action_in_range = (active_actions >= 0) & (active_actions < ACTION_COUNT)
    out_of_range_count = int((~action_in_range).sum())
    if out_of_range_count:
        problems.append(f"selected {out_of_range_count} out-of-range active actions")

    illegal_selected_count = 0
    if active_count and action_in_range.any():
        valid_offsets = np.flatnonzero(action_in_range)
        selected_invalid = active_invalid[valid_offsets, active_actions[action_in_range]]
        illegal_selected_count = int(selected_invalid.sum()) + out_of_range_count
    else:
        illegal_selected_count = out_of_range_count
    if illegal_selected_count:
        problems.append(f"selected {illegal_selected_count} illegal actions")

    illegal_weight_matrix = np.where(active_invalid, active_weights, 0.0)
    illegal_action_weight_mass_per_root = illegal_weight_matrix.sum(axis=1)
    legal_action_weight_mass_per_root = np.where(~active_invalid, active_weights, 0.0).sum(
        axis=1
    )
    illegal_mass_max = (
        float(illegal_action_weight_mass_per_root.max()) if active_count else 0.0
    )
    illegal_weight_max = float(illegal_weight_matrix.max()) if active_count else 0.0
    if illegal_mass_max > illegal_weight_atol:
        problems.append(
            "action_weights assign mass to illegal actions: "
            f"max row mass {illegal_mass_max:.6g}"
        )

    row_sums = active_weights.sum(axis=1) if active_count else np.asarray([], dtype=np.float64)
    return (
        {
            "actions_legal": illegal_selected_count == 0,
            "active_root_count": active_count,
            "inactive_root_count": int(root_count - active_count),
            "illegal_selected_action_count": illegal_selected_count,
            "out_of_range_selected_action_count": out_of_range_count,
            "illegal_action_weight_mass_max": illegal_mass_max,
            "illegal_action_weight_max": illegal_weight_max,
            "legal_action_weight_row_sum_min": (
                float(legal_action_weight_mass_per_root.min()) if active_count else None
            ),
            "legal_action_weight_row_sum_max": (
                float(legal_action_weight_mass_per_root.max()) if active_count else None
            ),
            "active_action_weight_row_sum_min": (
                float(row_sums.min()) if active_count else None
            ),
            "active_action_weight_row_sum_max": (
                float(row_sums.max()) if active_count else None
            ),
            "legal_action_count_per_root_min": (
                int((~active_invalid).sum(axis=1).min()) if active_count else 0
            ),
            "legal_action_count_per_root_max": (
                int((~active_invalid).sum(axis=1).max()) if active_count else 0
            ),
            "legal_action_count_per_root_sample": (
                (~active_invalid).sum(axis=1)[: min(active_count, 8)].astype(int).tolist()
                if active_count
                else []
            ),
        },
        problems,
    )


def _extract_mctx_root_values(output: Any) -> tuple[np.ndarray | None, str]:
    """Best-effort root-value extraction across MCTX versions."""

    search_tree = getattr(output, "search_tree", None)
    if search_tree is None:
        return None, "missing_search_tree"

    for name in ("node_values", "values", "raw_values"):
        value = getattr(search_tree, name, None)
        if value is None:
            continue
        value_shape = tuple(getattr(value, "shape", ()))
        value_ndim = len(value_shape)
        if value_ndim >= 2 and int(value_shape[0]) > 0:
            array = np.asarray(value[:, 0])
            return array.astype(np.float32, copy=False), f"search_tree.{name}[:,0]"
        if value_ndim == 1:
            array = np.asarray(value)
            return array.astype(np.float32, copy=False), f"search_tree.{name}"
        array = np.asarray(value)
        if array.ndim >= 2 and array.shape[0] > 0:
            return array[:, 0].astype(np.float32, copy=False), f"search_tree.{name}[:,0]"
        if array.ndim == 1:
            return array.astype(np.float32, copy=False), f"search_tree.{name}"

    try:
        summary = search_tree.summary()
    except Exception:
        summary = None
    if summary is not None:
        for name in ("value", "values", "root_value", "root_values"):
            value = getattr(summary, name, None)
            if value is not None:
                value_ndim = len(tuple(getattr(value, "shape", ())))
                if value_ndim == 1:
                    array = np.asarray(value)
                    return array.astype(np.float32, copy=False), f"summary.{name}"
                if value_ndim >= 2:
                    array = np.asarray(value[..., 0])
                    return (
                        array.astype(np.float32, copy=False),
                        f"summary.{name}[...,0]",
                    )
                array = np.asarray(value)
                if array.ndim == 1:
                    return array.astype(np.float32, copy=False), f"summary.{name}"
                if array.ndim >= 2:
                    return (
                        array[..., 0].astype(np.float32, copy=False),
                        f"summary.{name}[...,0]",
                    )

    return None, "unavailable"


def _materialize_mctx_search_payload(output: Any) -> tuple[int, bool]:
    """Materialize replay-needed MCTX payloads for payload-split profiling."""

    action_weights_host = np.asarray(output.action_weights)
    root_values_host, _root_value_source = _extract_mctx_root_values(output)
    byte_count = int(action_weights_host.nbytes)
    if root_values_host is not None:
        byte_count += int(root_values_host.nbytes)
    return byte_count, root_values_host is not None


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
    compact_visual_observation_source = str(
        config.get(
            "compact_visual_observation_source",
            COMPACT_VISUAL_OBSERVATION_SOURCE_HOST,
        )
    )
    if compact_visual_observation_source not in COMPACT_VISUAL_OBSERVATION_SOURCES:
        allowed_sources = ", ".join(sorted(COMPACT_VISUAL_OBSERVATION_SOURCES))
        raise ValueError(
            "compact_visual_observation_source must be one of "
            f"{allowed_sources}; got {compact_visual_observation_source!r}"
        )
    closed_loop_replay_index_enabled = bool(
        config.get("closed_loop_replay_index", True)
    )

    devices = jax.devices()
    backend = jax.default_backend()
    problems: list[str] = []
    if backend not in {"gpu", "cuda"}:
        problems.append(f"expected a GPU JAX backend, got {backend!r}")

    def linspace_matrix(rows: int, cols: int, scale: float) -> Any:
        return jnp.linspace(-scale, scale, rows * cols, dtype=jnp.float32).reshape(
            rows, cols
        )

    def linspace_kernel(height: int, width: int, in_channels: int, out_channels: int, scale: float) -> Any:
        return jnp.linspace(
            -scale,
            scale,
            height * width * in_channels * out_channels,
            dtype=jnp.float32,
        ).reshape(height, width, in_channels, out_channels)

    def update_resident_compact_visual_stack(
        device_stack: Any | None,
        renderer: Any,
        *,
        env_rows: int,
        players: int,
    ) -> Any:
        latest_device = getattr(renderer, "last_output_device", None)
        if latest_device is None:
            raise ValueError("resident compact visual mode requires renderer.last_output_device")
        _validate_resident_compact_visual_latest(
            latest_device,
            env_rows=int(env_rows),
            players=int(players),
        )
        expected_stack_shape = (int(env_rows), int(players), 4, 64, 64)
        if (
            device_stack is None
            or tuple(int(dim) for dim in getattr(device_stack, "shape", ()))
            != expected_stack_shape
        ):
            device_stack = jnp.zeros(expected_stack_shape, dtype=latest_device.dtype)
        return jnp.concatenate((device_stack[:, :, 1:], latest_device), axis=2)

    def zero_resident_compact_visual_stack(
        device_stack: Any | None,
        *,
        env_rows: int,
        players: int,
    ) -> Any:
        expected_stack_shape = (int(env_rows), int(players), 4, 64, 64)
        if (
            device_stack is None
            or tuple(int(dim) for dim in getattr(device_stack, "shape", ()))
            != expected_stack_shape
        ):
            return jnp.zeros(expected_stack_shape, dtype=jnp.uint8)
        return device_stack

    candidate_ego_rows = batch_size
    visual_encoder = observation_mode in {
        OBSERVATION_MODE_CURVYTRON_VISUAL_ROOT,
        OBSERVATION_MODE_CURVYTRON_HYBRID_COMPACT_VISUAL_SAMPLE,
    }
    compact_root_batch_for_validation = None
    compact_batch_for_validation = None
    compact_visual_manager_for_replay = None
    compact_search_contract: dict[str, Any] | None = None
    compact_replay_index_contract: dict[str, Any] | None = None
    compact_replay_index_timing_sec: float | None = None
    compact_loop_current_batch: Any | None = None
    compact_visual_resident_device_stack: Any | None = None
    compact_visual_resident_stack_update_times_sec: list[float] = []
    if observation_mode == OBSERVATION_MODE_CURVYTRON_VISUAL_ROOT:
        obs_dim = 4 * 64 * 64
        env_batch_size = batch_size
        candidate_ego_rows = env_batch_size * player_count
        root_batch_size = candidate_ego_rows
        legal_mask_profile = _legal_mask_profile(config)
        host_setup_started = time.perf_counter()

        env_index = np.arange(env_batch_size, dtype=np.uint16)[:, None, None, None, None]
        player_index = np.arange(player_count, dtype=np.uint16)[None, :, None, None, None]
        channel_index = np.arange(4, dtype=np.uint16)[None, None, :, None, None]
        y_index = np.arange(64, dtype=np.uint16)[None, None, None, :, None]
        x_index = np.arange(64, dtype=np.uint16)[None, None, None, None, :]
        obs_env_host = (
            (
                3 * x_index
                + 5 * y_index
                + 37 * channel_index
                + 11 * env_index
                + 53 * player_index
            )
            % 256
        ).astype(np.uint8, copy=False)

        ego_mask_host = np.ones((env_batch_size, player_count), dtype=np.bool_)
        legal_action_mask_host = np.broadcast_to(
            ego_mask_host[:, :, None],
            (env_batch_size, player_count, ACTION_COUNT),
        ).copy()
        if legal_mask_profile == "mixed_2of3":
            flat_legal = legal_action_mask_host.reshape(candidate_ego_rows, ACTION_COUNT)
            flat_row = np.arange(candidate_ego_rows, dtype=np.int64)
            flat_legal[flat_row % 3 == 0, 0] = False
            flat_legal[flat_row % 3 == 1, 1] = False
            flat_legal[flat_row % 3 == 2, 2] = False
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
        obs_host = obs_env_host.reshape(root_batch_size, 4, 64, 64)
        invalid_actions_host = ~legal_action_mask_host.reshape(
            root_batch_size,
            ACTION_COUNT,
        )
        active_root_mask_host = np.ones(root_batch_size, dtype=np.bool_)
        live_ego_rows = int(ego_mask_host.sum())
        expected_source_shape = (env_batch_size, player_count, 4, 64, 64)
        if obs_env_host.shape != expected_source_shape:
            raise AssertionError(
                f"expected obs_env shape {expected_source_shape}, got {obs_env_host.shape}"
            )
        if obs_host.shape != (root_batch_size, 4, 64, 64):
            raise AssertionError(
                "expected flattened visual root shape "
                f"{(root_batch_size, 4, 64, 64)}, got {obs_host.shape}"
            )
        if not np.any(~invalid_actions_host, axis=1).all():
            raise AssertionError("every visual Mctx root must have at least one legal action")
        host_observation_setup_sec = time.perf_counter() - host_setup_started

        transfer_started = time.perf_counter()
        obs = jax.device_put(obs_host)
        invalid_actions = jax.device_put(invalid_actions_host)
        obs.block_until_ready()
        invalid_actions.block_until_ready()
        host_to_device_transfer_sec = time.perf_counter() - transfer_started

        observation = {
            "mode": observation_mode,
            "source": "synthetic_curvytron_visual_policy_stack_shape",
            "build_path": "host_uint8_visual_stack_then_jax_device_put",
            "source_tensor_shape": list(obs_env_host.shape),
            "root_obs_shape": list(obs_host.shape),
            "root_obs_dtype": str(obs_host.dtype),
            "visual_stack_layout": "[B,P,4,64,64] -> [B*P,4,64,64]",
            "legal_mask_profile": legal_mask_profile,
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
                "Synthetic visual root benchmark for the current policy "
                "observation shape. It does not render real CurvyTron frames "
                "or prove training quality; it tests a device-resident JAX "
                "visual encoder plus Mctx search denominator."
            ),
        }
    elif observation_mode == OBSERVATION_MODE_CURVYTRON_HYBRID_COMPACT_VISUAL_SAMPLE:
        from curvyzero.env.vector_multiplayer_env import DEFAULT_BODY_CAPACITY
        from curvyzero.training.compact_policy_row_bridge import build_compact_root_batch_v1
        from curvyzero.training.source_state_batched_observation_profile import (
            CpuOracleBatchedObservationRenderer,
        )
        from curvyzero.training.source_state_hybrid_observation_profile import (
            HYBRID_STACK_STORAGE_DTYPE_UINT8,
            HybridBatchedObservationProfileManager,
            HybridBatchedStackProbeResult,
            HybridCompactBatch,
            HybridObservationProfileConfig,
        )

        if player_count != 2:
            raise ValueError(
                f"{OBSERVATION_MODE_CURVYTRON_HYBRID_COMPACT_VISUAL_SAMPLE} requires "
                f"player_count=2, got {player_count}"
            )
        obs_dim = 4 * 64 * 64
        env_batch_size = batch_size
        seed = _nonnegative_int(config, "seed", 123)
        rollout_steps = max(1, _nonnegative_int(config, "rollout_steps", 2))
        actor_count = min(
            env_batch_size,
            max(1, int(config.get("actor_count") or min(8, env_batch_size))),
        )
        native_actor_buffer = bool(config.get("native_actor_buffer", False))
        hybrid_refresh_observation_stack = bool(
            config.get("hybrid_refresh_observation_stack", True)
        )
        hybrid_borrow_single_actor_render_state = bool(
            config.get("hybrid_borrow_single_actor_render_state", False)
        )
        hybrid_persistent_compact_render_state_buffer = bool(
            config.get("hybrid_persistent_compact_render_state_buffer", False)
        )
        compact_root_copy_observation = bool(
            config.get("compact_root_copy_observation", True)
        )
        configured_body_capacity = int(config.get("body_capacity") or DEFAULT_BODY_CAPACITY)
        body_capacity = (
            DEFAULT_BODY_CAPACITY
            if configured_body_capacity <= 4
            else configured_body_capacity
        )

        class _CompactBatchCaptureProbe:
            backend_name = "mctx_hybrid_compact_visual_capture"
            semantics = "capture_hybrid_compact_batch_for_mctx_profile"

            def __init__(self) -> None:
                self.last_batch: HybridCompactBatch | None = None

            def run(self, observation: np.ndarray, action_mask: np.ndarray):
                del observation, action_mask
                raise AssertionError("compact visual sample must use run_compact_batch")

            def run_compact_batch(
                self,
                batch: HybridCompactBatch,
            ) -> HybridBatchedStackProbeResult:
                self.last_batch = batch
                return HybridBatchedStackProbeResult(
                    telemetry={
                        "total_sec": 0.0,
                        "compact_capture": True,
                        "root_count": int(batch.active_root_mask.shape[0]),
                        "active_root_count": int(np.count_nonzero(batch.active_root_mask)),
                    }
                )

        renderer_backend = str(
            config.get(
                "observation_renderer_backend",
                "jax_gpu_persistent_policy_framebuffer_profile",
            )
        )
        resident_compact_visual_observation = (
            compact_visual_observation_source == COMPACT_VISUAL_OBSERVATION_SOURCE_RESIDENT_GPU
        )
        compact_visual_resident_sync = bool(
            config.get("compact_visual_resident_sync", True)
        )
        persistent_renderer_async_device_only_profile = bool(
            config.get("persistent_renderer_async_device_only_profile", False)
        )
        persistent_vectorized_delta_pack_profile = bool(
            config.get("persistent_vectorized_delta_pack_profile", False)
        )
        if resident_compact_visual_observation and renderer_backend != (
            "jax_gpu_persistent_policy_framebuffer_profile"
        ):
            raise ValueError(
                "resident_gpu compact visual observation requires "
                "jax_gpu_persistent_policy_framebuffer_profile"
            )

        host_setup_started = time.perf_counter()
        capture_probe = _CompactBatchCaptureProbe()
        if renderer_backend == "cpu_oracle":
            observation_renderer = CpuOracleBatchedObservationRenderer()
        elif renderer_backend == "jax_gpu_persistent_policy_framebuffer_profile":
            from curvyzero.infra.modal.source_state_batched_observation_boundary_profile import (
                BONUS_RENDER_MODE_IDS,
                BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
                COMPUTE_H100,
                COMPUTE_L4_T4,
                RENDER_MODE_BROWSER_LINES,
                RENDER_MODE_IDS,
                RENDER_SURFACE_DIRECT_GRAY64,
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND,
                TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
                _make_jax_two_view_render_fn,
                _make_profile_observation_renderer,
                _validate_boundary_config,
            )

            normalized_compute = str(config.get("compute", "h100")).strip().lower()
            boundary_compute = COMPUTE_H100 if normalized_compute in {"h100", "gpu-h100"} else COMPUTE_L4_T4
            checked = _validate_boundary_config(
                np=np,
                config={
                    "batch_size": env_batch_size,
                    "compute": boundary_compute,
                    "seed": seed,
                    "body_capacity": body_capacity,
                    "steps": rollout_steps,
                    "warmup_steps": 0,
                    "hybrid_observation_canary": True,
                    "surface_stack_backend": TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
                    "observation_renderer_backend": (
                        SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                    ),
                    "render_surface": RENDER_SURFACE_DIRECT_GRAY64,
                    "allow_render_truncation": False,
                    "async_device_only_profile": persistent_renderer_async_device_only_profile,
                    "persistent_vectorized_delta_pack_profile": (
                        persistent_vectorized_delta_pack_profile
                    ),
                },
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

            observation_renderer = _make_profile_observation_renderer(
                jax=jax,
                jnp=jnp,
                np=np,
                checked=checked,
                render_fn_for_slots=render_fn_for_slots,
                bonus_render_mode_id=bonus_render_mode_id,
            )
        else:
            raise ValueError(
                "observation_renderer_backend must be 'cpu_oracle' or "
                "'jax_gpu_persistent_policy_framebuffer_profile'"
            )

        manager = HybridBatchedObservationProfileManager(
            HybridObservationProfileConfig(
                batch_size=env_batch_size,
                actor_count=actor_count,
                player_count=player_count,
                steps=rollout_steps,
                warmup_steps=0,
                seed=seed,
                body_capacity=body_capacity,
                stack_storage_dtype=HYBRID_STACK_STORAGE_DTYPE_UINT8,
                update_host_observation_stack=not resident_compact_visual_observation,
                refresh_observation_stack=hybrid_refresh_observation_stack,
                borrow_single_actor_render_state=hybrid_borrow_single_actor_render_state,
                persistent_compact_render_state_buffer=(
                    hybrid_persistent_compact_render_state_buffer
                ),
                materialize_scalar_timestep=False,
                native_actor_buffer=native_actor_buffer,
                pickle_payload=False,
            ),
            observation_renderer=observation_renderer,
            batched_stack_probe=capture_probe,
        )
        manager_init_sec = time.perf_counter() - host_setup_started

        rollout_step_times_sec: list[float] = []
        rollout_timings: list[dict[str, float]] = []
        step = None
        actions = np.full((env_batch_size, player_count), 1, dtype=np.int16)
        for _ in range(rollout_steps):
            step_started = time.perf_counter()
            step = manager.step(actions)
            resident_stack_update_sec = 0.0
            if resident_compact_visual_observation:
                resident_started = time.perf_counter()
                if hybrid_refresh_observation_stack:
                    compact_visual_resident_device_stack = update_resident_compact_visual_stack(
                        compact_visual_resident_device_stack,
                        observation_renderer,
                        env_rows=env_batch_size,
                        players=player_count,
                    )
                else:
                    compact_visual_resident_device_stack = zero_resident_compact_visual_stack(
                        compact_visual_resident_device_stack,
                        env_rows=env_batch_size,
                        players=player_count,
                    )
                if compact_visual_resident_sync:
                    compact_visual_resident_device_stack.block_until_ready()
                resident_stack_update_sec = time.perf_counter() - resident_started
                compact_visual_resident_stack_update_times_sec.append(resident_stack_update_sec)
            rollout_step_times_sec.append(time.perf_counter() - step_started)
            timing_sample = {
                key: float(value)
                for key, value in step.timings.items()
                if value
                and key
                in {
                    "actor_step_wall_sec",
                    "gather_merge_sec",
                    "observation_sec",
                    "renderer_render_sec",
                    "renderer_production_to_compact_sec",
                    "renderer_persistent_delta_pack_sec",
                    "renderer_host_to_device_sec",
                    "renderer_persistent_update_sec",
                    "renderer_device_render_sec",
                    "renderer_device_to_host_sec",
                    "renderer_stack_update_sec",
                    "stack_shift_sec",
                    "stack_latest_update_sec",
                    "compact_batch_build_sec",
                    "batched_stack_probe_wall_sec",
                    "batched_stack_probe_sec",
                }
            }
            if resident_stack_update_sec:
                timing_sample["resident_stack_update_sec"] = resident_stack_update_sec
            rollout_timings.append(timing_sample)

        if step is None or capture_probe.last_batch is None:
            raise AssertionError("hybrid compact visual sample did not produce a compact batch")
        compact_batch = capture_probe.last_batch
        compact_batch_for_validation = compact_batch
        compact_visual_manager_for_replay = manager

        root_build_started = time.perf_counter()
        compact_root_batch_for_validation = build_compact_root_batch_v1(
            compact_batch,
            search_lane="mctx_hybrid_compact_visual_sample",
            metadata={
                "observation_mode": observation_mode,
                "renderer_backend_name": manager.renderer_backend_name,
                "stack_storage_dtype": manager.stack_storage_dtype,
                "rollout_steps": rollout_steps,
                "actor_count": actor_count,
            },
            copy_observation=compact_root_copy_observation,
        )
        compact_root_build_sec = time.perf_counter() - root_build_started

        obs_host = np.asarray(compact_root_batch_for_validation.observation)
        if resident_compact_visual_observation:
            _validate_compact_visual_root_row_major_order(
                env_rows=compact_root_batch_for_validation.env_row,
                players=compact_root_batch_for_validation.player,
                env_row_count=int(compact_batch.observation.shape[0]),
                player_count=int(compact_batch.observation.shape[1]),
            )
        if obs_host.shape[1:] != (4, 64, 64):
            raise AssertionError(
                "expected compact visual root obs shape [root,4,64,64], got "
                f"{obs_host.shape}"
            )
        if obs_host.dtype != np.uint8:
            raise AssertionError(f"expected compact visual roots as uint8, got {obs_host.dtype}")

        root_batch_size = int(obs_host.shape[0])
        candidate_ego_rows = root_batch_size
        invalid_actions_host = ~np.asarray(
            compact_root_batch_for_validation.legal_mask,
            dtype=np.bool_,
        )
        active_root_mask_host = np.asarray(
            compact_root_batch_for_validation.active_root_mask,
            dtype=np.bool_,
        )
        if bool((~active_root_mask_host).any()):
            invalid_actions_host[~active_root_mask_host] = True
            invalid_actions_host[~active_root_mask_host, 0] = False
        live_ego_rows = int(active_root_mask_host.sum())
        host_observation_setup_sec = time.perf_counter() - host_setup_started

        transfer_started = time.perf_counter()
        if resident_compact_visual_observation:
            if compact_visual_resident_device_stack is None:
                raise AssertionError("resident compact visual stack was not initialized")
            obs = compact_visual_resident_device_stack.reshape(
                root_batch_size,
                4,
                64,
                64,
            )
        else:
            obs = jax.device_put(obs_host)
        invalid_actions = jax.device_put(invalid_actions_host)
        obs.block_until_ready()
        invalid_actions.block_until_ready()
        host_to_device_transfer_sec = time.perf_counter() - transfer_started

        observation = {
            "mode": observation_mode,
            "source": "HybridBatchedObservationProfileManager",
            "build_path": (
                (
                    "renderer_backed_HybridCompactBatch_sidecars_resident_gpu_stack"
                    if resident_compact_visual_observation
                    else "renderer_backed_HybridCompactBatch_CompactRootBatchV1_"
                    "then_jax_device_put"
                )
            ),
            "compact_visual_observation_source": compact_visual_observation_source,
            "root_observation_hot_source": (
                "resident_gpu_stack" if resident_compact_visual_observation else "host_stack"
            ),
            "root_observation_host_materialized_for_validation": True,
            "resident_stack_update_times_sec": compact_visual_resident_stack_update_times_sec,
            "source_tensor_shape": list(compact_batch.observation.shape),
            "root_obs_shape": list(obs_host.shape),
            "renderer_backend_name": manager.renderer_backend_name,
            "requested_renderer_backend": renderer_backend,
            "stack_storage_dtype": manager.stack_storage_dtype,
            "seed": seed,
            "rollout_steps": rollout_steps,
            "actor_count": actor_count,
            "native_actor_buffer": native_actor_buffer,
            "hybrid_refresh_observation_stack": hybrid_refresh_observation_stack,
            "hybrid_borrow_single_actor_render_state": (
                hybrid_borrow_single_actor_render_state
            ),
            "hybrid_persistent_compact_render_state_buffer": (
                hybrid_persistent_compact_render_state_buffer
            ),
            "compact_visual_resident_sync": compact_visual_resident_sync,
            "compact_root_copy_observation": compact_root_copy_observation,
            "env_batch_size": env_batch_size,
            "candidate_ego_rows": candidate_ego_rows,
            "live_ego_rows": live_ego_rows,
            "search_root_count": root_batch_size,
            "body_capacity": body_capacity,
            "configured_body_capacity": configured_body_capacity,
            "root_contract": compact_root_batch_for_validation.metadata,
            "compact_root_build_sec": compact_root_build_sec,
            "manager_contract": manager.contract(),
            "manager_init_sec": manager_init_sec,
            "rollout_step_times_sec": rollout_step_times_sec,
            "rollout_last_timings_sec": rollout_timings[-1] if rollout_timings else {},
            "rollout_timing_samples_sec": rollout_timings[: min(len(rollout_timings), 4)],
            "terminal_row_count": int(np.count_nonzero(compact_batch.terminal_row_mask)),
            "autoreset_row_count": int(np.count_nonzero(compact_batch.autoreset_row_mask)),
            "final_observation_row_count": int(
                np.count_nonzero(compact_batch.final_observation_row_mask)
            ),
            "shape_assertions": "passed",
            "note": (
                "Real renderer-backed hybrid compact visual roots. This validates "
                "CompactRootBatchV1 and runs Mctx search on real [B,2,4,64,64] "
                "stacks. It is profile-only and does not call train_muzero."
            ),
        }
    elif observation_mode == OBSERVATION_MODE_CURVYTRON_DEBUG:
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
        active_root_mask_host = np.ones(root_batch_size, dtype=np.bool_)
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
        active_root_mask_host = np.ones(int(obs_host.shape[0]), dtype=np.bool_)
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
        active_root_mask_host = np.ones(int(obs_host.shape[0]), dtype=np.bool_)
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
        active_root_mask_host = np.ones(root_batch_size, dtype=np.bool_)
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
        active_root_mask_host = np.asarray(mapping.row_mask, dtype=np.bool_)
        if (~active_root_mask_host).any():
            invalid_actions_host[~active_root_mask_host] = True
            invalid_actions_host[~active_root_mask_host, 0] = False
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

        active_rows = active_root_mask_host.astype(bool, copy=False)
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
        host_setup_started = time.perf_counter()
        obs_host = np.linspace(
            -1.0,
            1.0,
            root_batch_size * obs_dim,
            dtype=np.float32,
        ).reshape(root_batch_size, obs_dim)
        invalid_actions_host = np.zeros((root_batch_size, ACTION_COUNT), dtype=np.bool_)
        active_root_mask_host = np.ones(root_batch_size, dtype=np.bool_)
        host_observation_setup_sec = time.perf_counter() - host_setup_started
        transfer_started = time.perf_counter()
        obs = jax.device_put(obs_host)
        invalid_actions = jax.device_put(invalid_actions_host)
        obs.block_until_ready()
        invalid_actions.block_until_ready()
        host_to_device_transfer_sec = time.perf_counter() - transfer_started
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

    active_root_mask_host = np.asarray(active_root_mask_host, dtype=np.bool_)
    if active_root_mask_host.shape != (root_batch_size,):
        raise AssertionError(
            "active_root_mask_host must have shape "
            f"{(root_batch_size,)}, got {active_root_mask_host.shape}"
        )
    active_root_count = int(active_root_mask_host.sum())
    if active_root_count < 1:
        raise ValueError("MCTX benchmark requires at least one active root")
    if not np.any(~invalid_actions_host[active_root_mask_host], axis=1).all():
        raise AssertionError("every active Mctx root must have at least one legal action")
    observation["active_root_mask_shape"] = list(active_root_mask_host.shape)
    observation["active_root_count"] = active_root_count
    observation["inactive_root_count"] = int(root_batch_size - active_root_count)

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
        "action_embed": linspace_matrix(ACTION_COUNT, hidden_dim, 0.10),
        "dynamics_w": linspace_matrix(hidden_dim, hidden_dim, 0.025),
        "dynamics_b": jnp.linspace(0.01, -0.01, hidden_dim, dtype=jnp.float32),
        "policy_w": linspace_matrix(hidden_dim, ACTION_COUNT, 0.07),
        "policy_b": jnp.array([0.0, 0.03, -0.01], dtype=jnp.float32),
        "value_w": jnp.linspace(-0.04, 0.04, hidden_dim, dtype=jnp.float32),
        "reward_w": jnp.linspace(0.03, -0.03, hidden_dim, dtype=jnp.float32),
    }
    if visual_encoder:
        visual_channels = _positive_int(config, "visual_channels", 8)
        params.update(
            {
                "visual_conv1": linspace_kernel(3, 3, 4, visual_channels, 0.035),
                "visual_conv1_b": jnp.linspace(
                    -0.01,
                    0.01,
                    visual_channels,
                    dtype=jnp.float32,
                ),
                "visual_conv2": linspace_kernel(
                    3,
                    3,
                    visual_channels,
                    visual_channels,
                    0.025,
                ),
                "visual_conv2_b": jnp.linspace(
                    0.01,
                    -0.01,
                    visual_channels,
                    dtype=jnp.float32,
                ),
                "visual_projection_w": linspace_matrix(visual_channels, hidden_dim, 0.08),
                "visual_projection_b": jnp.linspace(
                    -0.02,
                    0.02,
                    hidden_dim,
                    dtype=jnp.float32,
                ),
            }
        )
    else:
        params.update(
            {
                "representation_w": linspace_matrix(obs_dim, hidden_dim, 0.08),
                "representation_b": jnp.linspace(-0.02, 0.02, hidden_dim, dtype=jnp.float32),
            }
        )

    def representation(params: dict[str, Any], obs_batch: Any) -> Any:
        if visual_encoder:
            x = obs_batch.astype(jnp.float32) * jnp.float32(1.0 / 255.0)
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = jax.lax.conv_general_dilated(
                x,
                params["visual_conv1"],
                window_strides=(2, 2),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            x = jax.nn.relu(x + params["visual_conv1_b"])
            x = jax.lax.conv_general_dilated(
                x,
                params["visual_conv2"],
                window_strides=(2, 2),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            x = jax.nn.relu(x + params["visual_conv2_b"])
            x = jnp.mean(x, axis=(1, 2))
            return jnp.tanh(x @ params["visual_projection_w"] + params["visual_projection_b"])
        return jnp.tanh(obs_batch @ params["representation_w"] + params["representation_b"])

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
        hidden = representation(params, obs)
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

    end_to_end_steady_times_sec: list[float] = []
    end_to_end_h2d_times_sec: list[float] = []
    end_to_end_search_times_sec: list[float] = []
    end_to_end_d2h_times_sec: list[float] = []
    for run_index in range(steady_runs):
        e2e_started = time.perf_counter()
        transfer_started = time.perf_counter()
        e2e_obs = jax.device_put(obs_host)
        e2e_invalid_actions = jax.device_put(invalid_actions_host)
        e2e_obs.block_until_ready()
        e2e_invalid_actions.block_until_ready()
        end_to_end_h2d_times_sec.append(time.perf_counter() - transfer_started)

        search_started = time.perf_counter()
        e2e_output = run_search(
            params,
            jax.random.PRNGKey(2000 + run_index),
            e2e_obs,
            e2e_invalid_actions,
            num_simulations=num_simulations,
            max_depth=max_depth,
        )
        e2e_output.action_weights.block_until_ready()
        end_to_end_search_times_sec.append(time.perf_counter() - search_started)

        d2h_started = time.perf_counter()
        np.asarray(e2e_output.action)
        np.asarray(e2e_output.action_weights)
        end_to_end_d2h_times_sec.append(time.perf_counter() - d2h_started)
        end_to_end_steady_times_sec.append(time.perf_counter() - e2e_started)

    steady_median_sec = statistics.median(steady_times_sec)
    steady_min_sec = min(steady_times_sec)
    steady_max_sec = max(steady_times_sec)
    end_to_end_steady_median_sec = statistics.median(end_to_end_steady_times_sec)
    end_to_end_h2d_median_sec = statistics.median(end_to_end_h2d_times_sec)
    end_to_end_search_median_sec = statistics.median(end_to_end_search_times_sec)
    end_to_end_d2h_median_sec = statistics.median(end_to_end_d2h_times_sec)
    decisions_per_sec_median = root_batch_size / steady_median_sec
    simulations_per_sec_median = (root_batch_size * num_simulations) / steady_median_sec
    active_decisions_per_sec_median = active_root_count / steady_median_sec
    end_to_end_decisions_per_sec_median = root_batch_size / end_to_end_steady_median_sec
    end_to_end_active_decisions_per_sec_median = (
        active_root_count / end_to_end_steady_median_sec
    )
    end_to_end_simulations_per_sec_median = (
        root_batch_size * num_simulations
    ) / end_to_end_steady_median_sec
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
        "active_search_root_count": active_root_count,
        "inactive_search_root_count": int(root_batch_size - active_root_count),
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
    active_root_mask_np = np.asarray(active_root_mask_host, dtype=np.bool_)
    active_row_sums = row_sums[active_root_mask_np]
    normalized_weights = bool(np.allclose(active_row_sums, 1.0, atol=1e-5))
    invalid_actions_np = np.asarray(invalid_actions_host, dtype=np.bool_)
    legality_summary, legality_problems = _mctx_legality_summary(
        actions,
        action_weights,
        invalid_actions_np,
        active_root_mask_np,
    )
    if not finite_weights:
        problems.append("action_weights contains non-finite values")
    if not normalized_weights:
        problems.append("active action_weights rows do not sum to 1 within atol=1e-5")
    problems.extend(legality_problems)

    active_indices = np.flatnonzero(active_root_mask_np).astype(np.int32, copy=False)
    compact_sample_count = min(int(active_indices.size), 16)
    root_values, root_value_source = _extract_mctx_root_values(output)
    compact_root_values_sample = (
        None
        if root_values is None
        else root_values[active_indices[:compact_sample_count]].astype(float).tolist()
    )
    if compact_root_batch_for_validation is not None:
        if root_values is None:
            problems.append("compact search validation requires root values")
        else:
            from curvyzero.training.compact_policy_row_bridge import (
                build_compact_replay_index_rows_v1_from_search_result,
                validate_compact_search_result_v1,
            )

            compact_search_result = validate_compact_search_result_v1(
                compact_root_batch_for_validation,
                selected_action=actions[active_indices],
                visit_policy=action_weights[active_indices],
                root_value=root_values[active_indices],
                search_impl="mctx_gumbel_muzero_policy",
                num_simulations=num_simulations,
                metadata={
                    "observation_mode": observation_mode,
                    "root_value_source": root_value_source,
                },
            )
            compact_search_contract = {
                "metadata": compact_search_result.metadata,
                "root_index_sample": compact_search_result.root_index[
                    :compact_sample_count
                ].astype(int).tolist(),
                "selected_action_sample": compact_search_result.selected_action[
                    :compact_sample_count
                ].astype(int).tolist(),
                "visit_policy_sample": compact_search_result.visit_policy[
                    :compact_sample_count
                ].astype(float).tolist(),
            }
            if (
                compact_batch_for_validation is not None
                and compact_visual_manager_for_replay is not None
            ):
                replay_started = time.perf_counter()
                next_joint_action = np.full(
                    (
                        int(compact_batch_for_validation.observation.shape[0]),
                        int(compact_batch_for_validation.observation.shape[1]),
                    ),
                    1,
                    dtype=np.int16,
                )
                next_joint_action[
                    compact_search_result.env_row.astype(np.int64, copy=False),
                    compact_search_result.player.astype(np.int64, copy=False),
                ] = compact_search_result.selected_action
                next_step = compact_visual_manager_for_replay.step(next_joint_action)
                next_resident_stack_update_sec = 0.0
                if resident_compact_visual_observation:
                    resident_started = time.perf_counter()
                    if hybrid_refresh_observation_stack:
                        compact_visual_resident_device_stack = update_resident_compact_visual_stack(
                            compact_visual_resident_device_stack,
                            observation_renderer,
                            env_rows=int(compact_batch_for_validation.observation.shape[0]),
                            players=int(compact_batch_for_validation.observation.shape[1]),
                        )
                    else:
                        compact_visual_resident_device_stack = zero_resident_compact_visual_stack(
                            compact_visual_resident_device_stack,
                            env_rows=int(compact_batch_for_validation.observation.shape[0]),
                            players=int(compact_batch_for_validation.observation.shape[1]),
                        )
                    if compact_visual_resident_sync:
                        compact_visual_resident_device_stack.block_until_ready()
                    next_resident_stack_update_sec = time.perf_counter() - resident_started
                compact_loop_current_batch = next_step.compact_batch
                next_final_mask = (
                    np.zeros_like(next_step.done, dtype=np.bool_)
                    if next_step.compact_batch is None
                    else next_step.compact_batch.final_observation_row_mask
                )
                replay_index_rows = build_compact_replay_index_rows_v1_from_search_result(
                    compact_batch_for_validation,
                    compact_root_batch_for_validation,
                    compact_search_result,
                    record_index=0,
                    next_joint_action=next_joint_action,
                    next_reward=next_step.reward,
                    next_done=next_step.done,
                    next_final_reward_map=next_step.reward,
                    next_final_observation_row_mask=next_final_mask,
                    policy_source="mctx_hybrid_compact_visual_sample",
                    metadata={
                        "observation_mode": observation_mode,
                        "search_impl": "mctx_gumbel_muzero_policy",
                    },
                )
                compact_replay_index_timing_sec = time.perf_counter() - replay_started
                replay_next_step_timings = {
                    key: float(value)
                    for key, value in next_step.timings.items()
                    if value
                    and key
                    in {
                        "actor_step_wall_sec",
                        "actor_env_runtime_sec",
                        "actor_compact_write_sec",
                        "actor_render_state_write_sec",
                        "actor_render_state_write_visual_trail_sec",
                        "actor_render_state_write_player_sec",
                        "actor_render_state_write_bonus_sec",
                        "actor_render_state_write_other_sec",
                        "gather_merge_sec",
                        "observation_sec",
                        "renderer_render_sec",
                        "renderer_stack_update_sec",
                        "stack_shift_sec",
                        "stack_latest_update_sec",
                        "compact_batch_build_sec",
                        "batched_stack_probe_wall_sec",
                        "batched_stack_probe_sec",
                    }
                }
                if next_resident_stack_update_sec:
                    replay_next_step_timings["resident_stack_update_sec"] = (
                        next_resident_stack_update_sec
                    )
                compact_replay_index_contract = {
                    "metadata": replay_index_rows.metadata,
                    "index_row_count": int(replay_index_rows.action.shape[0]),
                    "action_sample": replay_index_rows.action[
                        :compact_sample_count
                    ].astype(int).tolist(),
                    "policy_target_sample": replay_index_rows.policy_target[
                        : min(compact_sample_count, replay_index_rows.policy_target.shape[0])
                    ].astype(float).tolist(),
                    "timing_sec": compact_replay_index_timing_sec,
                    "next_step_timings_sec": replay_next_step_timings,
                }

    output_d2h_median_sec = statistics.median(device_to_host_action_weights_times_sec)
    steady_search_plus_h2d_plus_policy_d2h_median_sec = steady_median_sec + (
        transfer_steady_median_sec or 0.0
    ) + output_d2h_median_sec
    steady_search_plus_h2d_plus_policy_d2h_roots_per_sec = (
        root_batch_size / steady_search_plus_h2d_plus_policy_d2h_median_sec
    )
    host_setup_plus_fresh_boundary_sec = (
        host_observation_setup_sec + end_to_end_steady_median_sec
    )
    host_setup_plus_fresh_boundary_active_decisions_per_sec = (
        active_root_count / host_setup_plus_fresh_boundary_sec
    )
    closed_one_step_search_replay_edge_sec = None
    closed_one_step_search_replay_edge_active_decisions_per_sec = None
    if compact_replay_index_timing_sec is not None:
        closed_one_step_search_replay_edge_sec = (
            end_to_end_steady_median_sec + compact_replay_index_timing_sec
        )
        closed_one_step_search_replay_edge_active_decisions_per_sec = (
            active_root_count / closed_one_step_search_replay_edge_sec
        )
    closed_compact_loop_contract: dict[str, Any] | None = None
    closed_loop_steps = _nonnegative_int(config, "closed_loop_steps", 0)
    if closed_loop_steps:
        if compact_visual_manager_for_replay is None or compact_loop_current_batch is None:
            problems.append("closed_loop_steps requires compact visual replay state")
        else:
            closed_loop_action_only_profile = bool(
                config.get("closed_loop_action_only_profile", False)
            )
            closed_loop_deferred_payload_profile = bool(
                config.get("closed_loop_deferred_payload_profile", False)
            )
            closed_loop_overlap_payload_profile = bool(
                config.get("closed_loop_overlap_payload_profile", False)
            )
            payload_profile_count = sum(
                (
                    closed_loop_action_only_profile,
                    closed_loop_deferred_payload_profile,
                    closed_loop_overlap_payload_profile,
                )
            )
            if payload_profile_count > 1:
                raise ValueError(
                    "closed_loop_action_only_profile and "
                    "closed_loop_deferred_payload_profile and "
                    "closed_loop_overlap_payload_profile are mutually exclusive"
                )
            if closed_loop_action_only_profile and closed_loop_replay_index_enabled:
                raise ValueError(
                    "closed_loop_action_only_profile requires "
                    "closed_loop_replay_index=False; it skips root-value/policy "
                    "materialization and is not a replay-valid lane"
                )
            if closed_loop_deferred_payload_profile and closed_loop_replay_index_enabled:
                raise ValueError(
                    "closed_loop_deferred_payload_profile requires "
                    "closed_loop_replay_index=False; replay rows are not built "
                    "until a later resident/chunked payload design exists"
                )
            if closed_loop_overlap_payload_profile and closed_loop_replay_index_enabled:
                raise ValueError(
                    "closed_loop_overlap_payload_profile requires "
                    "closed_loop_replay_index=False; replay rows are not built "
                    "until a later overlapped payload commit design exists"
                )
            from curvyzero.training.compact_policy_row_bridge import (
                build_compact_replay_index_rows_v1_from_search_result,
                build_compact_root_batch_v1,
                validate_compact_search_result_v1,
            )

            loop_records: list[dict[str, Any]] = []
            deferred_payload_outputs: list[Any] = []
            overlap_payload_executor = (
                ThreadPoolExecutor(max_workers=1)
                if closed_loop_overlap_payload_profile
                else None
            )
            overlapped_search_payload_wait_sec = 0.0
            overlapped_search_payload_bytes = 0
            overlapped_search_payload_count = 0
            loop_total_active_roots = 0
            loop_total_started = time.perf_counter()
            loop_batch = compact_loop_current_batch
            for loop_index in range(closed_loop_steps):
                if loop_batch is None:
                    break

                loop_started = time.perf_counter()
                loop_root_started = time.perf_counter()
                loop_root_batch = build_compact_root_batch_v1(
                    loop_batch,
                    search_lane="mctx_hybrid_compact_visual_closed_loop",
                    metadata={
                        "observation_mode": observation_mode,
                        "loop_index": loop_index,
                    },
                    copy_observation=bool(config.get("compact_root_copy_observation", True)),
                )
                loop_root_build_sec = time.perf_counter() - loop_root_started
                loop_root_sidecar_started = time.perf_counter()
                if resident_compact_visual_observation:
                    _validate_compact_visual_root_row_major_order(
                        env_rows=loop_root_batch.env_row,
                        players=loop_root_batch.player,
                        env_row_count=int(loop_batch.observation.shape[0]),
                        player_count=int(loop_batch.observation.shape[1]),
                    )
                loop_obs_host = np.asarray(loop_root_batch.observation)
                loop_invalid_host = ~np.asarray(loop_root_batch.legal_mask, dtype=np.bool_)
                loop_active_mask = np.asarray(
                    loop_root_batch.active_root_mask,
                    dtype=np.bool_,
                )
                if bool((~loop_active_mask).any()):
                    loop_invalid_host[~loop_active_mask] = True
                    loop_invalid_host[~loop_active_mask, 0] = False
                loop_active_indices = np.flatnonzero(loop_active_mask).astype(
                    np.int32,
                    copy=False,
                )
                loop_active_count = int(loop_active_indices.size)
                if loop_active_count < 1:
                    break
                loop_root_sidecar_sec = time.perf_counter() - loop_root_sidecar_started

                loop_h2d_started = time.perf_counter()
                if resident_compact_visual_observation:
                    if compact_visual_resident_device_stack is None:
                        raise AssertionError("resident compact visual stack was not initialized")
                    loop_obs = compact_visual_resident_device_stack.reshape(
                        int(loop_root_batch.observation.shape[0]),
                        4,
                        64,
                        64,
                    )
                else:
                    loop_obs = jax.device_put(loop_obs_host)
                loop_invalid = jax.device_put(loop_invalid_host)
                loop_obs.block_until_ready()
                loop_invalid.block_until_ready()
                loop_h2d_sec = time.perf_counter() - loop_h2d_started

                loop_search_started = time.perf_counter()
                loop_output = run_search(
                    params,
                    jax.random.PRNGKey(3000 + loop_index),
                    loop_obs,
                    loop_invalid,
                    num_simulations=num_simulations,
                    max_depth=max_depth,
                )
                if (
                    closed_loop_action_only_profile
                    or closed_loop_deferred_payload_profile
                    or closed_loop_overlap_payload_profile
                ):
                    loop_output.action.block_until_ready()
                else:
                    loop_output.action_weights.block_until_ready()
                loop_search_sec = time.perf_counter() - loop_search_started

                loop_d2h_started = time.perf_counter()
                loop_actions = np.asarray(loop_output.action)
                loop_action_weights = (
                    None
                    if (
                        closed_loop_action_only_profile
                        or closed_loop_deferred_payload_profile
                        or closed_loop_overlap_payload_profile
                    )
                    else np.asarray(loop_output.action_weights)
                )
                loop_d2h_sec = time.perf_counter() - loop_d2h_started
                loop_root_value_extract_sec = 0.0
                loop_search_result_validate_sec = 0.0
                loop_overlap_payload_wait_sec = 0.0
                loop_overlap_future = None
                if (
                    closed_loop_action_only_profile
                    or closed_loop_deferred_payload_profile
                    or closed_loop_overlap_payload_profile
                ):
                    if closed_loop_deferred_payload_profile:
                        deferred_payload_outputs.append(loop_output)
                    if closed_loop_overlap_payload_profile:
                        if overlap_payload_executor is None:
                            raise AssertionError("overlap payload executor missing")
                        loop_overlap_future = overlap_payload_executor.submit(
                            _materialize_mctx_search_payload,
                            loop_output,
                        )
                    loop_joint_action_started = time.perf_counter()
                    loop_joint_action = np.full(
                        (
                            int(loop_batch.observation.shape[0]),
                            int(loop_batch.observation.shape[1]),
                        ),
                        1,
                        dtype=np.int16,
                    )
                    selected_action = loop_actions[loop_active_indices]
                    env_rows_for_action = np.asarray(
                        loop_root_batch.env_row,
                        dtype=np.int64,
                    )[loop_active_indices]
                    players_for_action = np.asarray(
                        loop_root_batch.player,
                        dtype=np.int64,
                    )[loop_active_indices]
                    if bool(
                        (
                            loop_invalid_host[
                                loop_active_indices,
                                selected_action.astype(np.int64, copy=False),
                            ]
                        ).any()
                    ):
                        raise AssertionError(
                            "closed_loop_action_only_profile selected an illegal action"
                        )
                    loop_joint_action[
                        env_rows_for_action,
                        players_for_action,
                    ] = selected_action
                    loop_joint_action_build_sec = (
                        time.perf_counter() - loop_joint_action_started
                    )
                else:
                    loop_root_value_started = time.perf_counter()
                    loop_root_values, loop_root_value_source = _extract_mctx_root_values(
                        loop_output
                    )
                    loop_root_value_extract_sec = time.perf_counter() - loop_root_value_started
                    if loop_root_values is None:
                        problems.append("closed compact loop requires MCTX root values")
                        break

                    loop_validate_started = time.perf_counter()
                    loop_search_result = validate_compact_search_result_v1(
                        loop_root_batch,
                        selected_action=loop_actions[loop_active_indices],
                        visit_policy=loop_action_weights[loop_active_indices],
                        root_value=loop_root_values[loop_active_indices],
                        search_impl="mctx_gumbel_muzero_policy",
                        num_simulations=num_simulations,
                        metadata={
                            "observation_mode": observation_mode,
                            "loop_index": loop_index,
                            "root_value_source": loop_root_value_source,
                        },
                    )
                    loop_search_result_validate_sec = (
                        time.perf_counter() - loop_validate_started
                    )
                    loop_joint_action_started = time.perf_counter()
                    loop_joint_action = np.full(
                        (
                            int(loop_batch.observation.shape[0]),
                            int(loop_batch.observation.shape[1]),
                        ),
                        1,
                        dtype=np.int16,
                    )
                    loop_joint_action[
                        loop_search_result.env_row.astype(np.int64, copy=False),
                        loop_search_result.player.astype(np.int64, copy=False),
                    ] = loop_search_result.selected_action
                    loop_joint_action_build_sec = (
                        time.perf_counter() - loop_joint_action_started
                    )

                loop_step_started = time.perf_counter()
                loop_next_step = compact_visual_manager_for_replay.step(loop_joint_action)
                loop_resident_stack_update_sec = 0.0
                if resident_compact_visual_observation:
                    resident_started = time.perf_counter()
                    if hybrid_refresh_observation_stack:
                        compact_visual_resident_device_stack = update_resident_compact_visual_stack(
                            compact_visual_resident_device_stack,
                            observation_renderer,
                            env_rows=int(loop_batch.observation.shape[0]),
                            players=int(loop_batch.observation.shape[1]),
                        )
                    else:
                        compact_visual_resident_device_stack = zero_resident_compact_visual_stack(
                            compact_visual_resident_device_stack,
                            env_rows=int(loop_batch.observation.shape[0]),
                            players=int(loop_batch.observation.shape[1]),
                        )
                    if compact_visual_resident_sync:
                        compact_visual_resident_device_stack.block_until_ready()
                    loop_resident_stack_update_sec = time.perf_counter() - resident_started
                loop_step_sec = time.perf_counter() - loop_step_started
                if loop_overlap_future is not None:
                    overlap_wait_started = time.perf_counter()
                    payload_bytes, root_values_present = loop_overlap_future.result()
                    loop_overlap_payload_wait_sec = (
                        time.perf_counter() - overlap_wait_started
                    )
                    overlapped_search_payload_wait_sec += loop_overlap_payload_wait_sec
                    overlapped_search_payload_bytes += int(payload_bytes)
                    overlapped_search_payload_count += 1
                    if not root_values_present:
                        problems.append(
                            "overlap payload profile could not extract MCTX root values"
                        )
                loop_next_final_mask = (
                    np.zeros_like(loop_next_step.done, dtype=np.bool_)
                    if loop_next_step.compact_batch is None
                    else loop_next_step.compact_batch.final_observation_row_mask
                )

                loop_replay_started = time.perf_counter()
                loop_index_row_count = 0
                if closed_loop_replay_index_enabled:
                    if (
                        closed_loop_action_only_profile
                        or closed_loop_deferred_payload_profile
                        or closed_loop_overlap_payload_profile
                    ):
                        raise AssertionError(
                            "action-only/deferred/overlap payload profiles cannot "
                            "build replay rows"
                        )
                    loop_replay_rows = build_compact_replay_index_rows_v1_from_search_result(
                        loop_batch,
                        loop_root_batch,
                        loop_search_result,
                        record_index=loop_index + 1,
                        next_joint_action=loop_joint_action,
                        next_reward=loop_next_step.reward,
                        next_done=loop_next_step.done,
                        next_final_reward_map=loop_next_step.reward,
                        next_final_observation_row_mask=loop_next_final_mask,
                        policy_source="mctx_hybrid_compact_visual_closed_loop",
                        metadata={
                            "observation_mode": observation_mode,
                            "loop_index": loop_index,
                            "search_impl": "mctx_gumbel_muzero_policy",
                        },
                    )
                    loop_index_row_count = int(loop_replay_rows.action.shape[0])
                loop_replay_sec = time.perf_counter() - loop_replay_started
                loop_total_active_roots += loop_active_count
                next_step_timings = {
                    key: float(value)
                    for key, value in loop_next_step.timings.items()
                    if value
                    and key
                    in {
                        "actor_step_wall_sec",
                        "actor_env_public_prepare_sec",
                        "actor_env_runtime_sec",
                        "actor_env_post_runtime_bookkeeping_sec",
                        "actor_env_reward_sec",
                        "actor_env_final_observation_sec",
                        "actor_env_public_info_sec",
                        "actor_env_batch_pack_sec",
                        "actor_payload_copy_sec",
                        "actor_compact_write_sec",
                        "actor_render_state_write_sec",
                        "actor_render_state_write_visual_trail_sec",
                        "actor_render_state_write_player_sec",
                        "actor_render_state_write_bonus_sec",
                        "actor_render_state_write_other_sec",
                        "actor_autoreset_sec",
                        "gather_merge_sec",
                        "observation_sec",
                        "renderer_render_sec",
                        "renderer_production_to_compact_sec",
                        "renderer_persistent_delta_pack_sec",
                        "renderer_host_to_device_sec",
                        "renderer_persistent_update_sec",
                        "renderer_device_render_sec",
                        "renderer_device_to_host_sec",
                        "renderer_stack_update_sec",
                        "stack_shift_sec",
                        "stack_latest_update_sec",
                        "compact_batch_build_sec",
                        "batched_stack_probe_wall_sec",
                        "batched_stack_probe_sec",
                    }
                }
                if loop_resident_stack_update_sec:
                    next_step_timings["resident_stack_update_sec"] = loop_resident_stack_update_sec
                loop_records.append(
                    {
                        "loop_index": loop_index,
                        "active_roots": loop_active_count,
                        "root_build_sec": loop_root_build_sec,
                        "root_sidecar_sec": loop_root_sidecar_sec,
                        "h2d_sec": loop_h2d_sec,
                        "search_sec": loop_search_sec,
                        "d2h_sec": loop_d2h_sec,
                        "root_value_extract_sec": loop_root_value_extract_sec,
                        "search_result_validate_sec": loop_search_result_validate_sec,
                        "overlapped_search_payload_wait_sec": (
                            loop_overlap_payload_wait_sec
                        ),
                        "joint_action_build_sec": loop_joint_action_build_sec,
                        "env_step_sec": loop_step_sec,
                        "replay_index_sec": loop_replay_sec,
                        "total_sec": time.perf_counter() - loop_started,
                        "index_row_count": loop_index_row_count,
                        "next_step_timings_sec": next_step_timings,
                    }
                )
                loop_batch = loop_next_step.compact_batch

            loop_action_total_sec = time.perf_counter() - loop_total_started
            deferred_search_payload_flush_sec = 0.0
            deferred_search_payload_bytes = 0
            deferred_search_payload_count = 0
            if closed_loop_deferred_payload_profile:
                flush_started = time.perf_counter()
                for loop_output in deferred_payload_outputs:
                    action_weights_host = np.asarray(loop_output.action_weights)
                    root_values_host, _root_value_source = _extract_mctx_root_values(
                        loop_output
                    )
                    deferred_search_payload_bytes += int(action_weights_host.nbytes)
                    if root_values_host is not None:
                        deferred_search_payload_bytes += int(root_values_host.nbytes)
                    else:
                        problems.append(
                            "deferred payload profile could not extract MCTX root values"
                        )
                    deferred_search_payload_count += 1
                deferred_search_payload_flush_sec = time.perf_counter() - flush_started
            if overlap_payload_executor is not None:
                overlap_payload_executor.shutdown(wait=True)
            loop_total_sec = time.perf_counter() - loop_total_started
            bucket_names = (
                "root_build_sec",
                "root_sidecar_sec",
                "h2d_sec",
                "search_sec",
                "d2h_sec",
                "root_value_extract_sec",
                "search_result_validate_sec",
                "overlapped_search_payload_wait_sec",
                "joint_action_build_sec",
                "env_step_sec",
                "replay_index_sec",
            )
            loop_bucket_totals_sec = {
                name: float(sum(record[name] for record in loop_records))
                for name in bucket_names
            }
            loop_bucket_totals_sec["deferred_search_payload_flush_sec"] = float(
                deferred_search_payload_flush_sec
            )
            next_step_bucket_totals_sec: dict[str, float] = {}
            for record in loop_records:
                for name, value in record["next_step_timings_sec"].items():
                    next_step_bucket_totals_sec[name] = (
                        next_step_bucket_totals_sec.get(name, 0.0) + float(value)
                    )
            measured_bucket_sum_sec = float(sum(loop_bucket_totals_sec.values()))
            residual_sec = float(loop_total_sec - measured_bucket_sum_sec)
            bucket_fraction_of_total = {
                name: (
                    None
                    if loop_total_sec <= 0.0
                    else float(value) / float(loop_total_sec)
                )
                for name, value in loop_bucket_totals_sec.items()
            }
            bucket_roots_per_sec = {
                name: (
                    None
                    if value <= 0.0 or loop_total_active_roots == 0
                    else float(loop_total_active_roots) / float(value)
                )
                for name, value in loop_bucket_totals_sec.items()
            }
            slowest_bucket = (
                None
                if not loop_bucket_totals_sec
                else max(loop_bucket_totals_sec, key=loop_bucket_totals_sec.get)
            )
            closed_compact_loop_contract = {
                "requested_steps": closed_loop_steps,
                "completed_steps": len(loop_records),
                "replay_index_enabled": closed_loop_replay_index_enabled,
                "action_only_profile": closed_loop_action_only_profile,
                "deferred_payload_profile": closed_loop_deferred_payload_profile,
                "overlap_payload_profile": closed_loop_overlap_payload_profile,
                "total_active_roots": loop_total_active_roots,
                "total_sec": loop_total_sec,
                "action_loop_total_sec": loop_action_total_sec,
                "deferred_search_payload_flush_sec": deferred_search_payload_flush_sec,
                "deferred_search_payload_bytes": deferred_search_payload_bytes,
                "deferred_search_payload_count": deferred_search_payload_count,
                "overlapped_search_payload_wait_sec": (
                    overlapped_search_payload_wait_sec
                ),
                "overlapped_search_payload_bytes": overlapped_search_payload_bytes,
                "overlapped_search_payload_count": overlapped_search_payload_count,
                "active_roots_per_sec": (
                    None
                    if loop_total_active_roots == 0
                    else loop_total_active_roots / loop_total_sec
                ),
                "bucket_totals_sec": loop_bucket_totals_sec,
                "next_step_bucket_totals_sec": next_step_bucket_totals_sec,
                "bucket_fraction_of_total": bucket_fraction_of_total,
                "bucket_active_roots_per_sec": bucket_roots_per_sec,
                "measured_bucket_sum_sec": measured_bucket_sum_sec,
                "residual_sec": residual_sec,
                "residual_fraction_of_total": (
                    None if loop_total_sec <= 0.0 else residual_sec / loop_total_sec
                ),
                "slowest_bucket": slowest_bucket,
                "step_records": loop_records,
            }

    closed_compact_loop_active_roots_per_sec = (
        None
        if closed_compact_loop_contract is None
        else closed_compact_loop_contract["active_roots_per_sec"]
    )
    compact_search_service_profile = _mctx_compact_visual_search_service_profile_row(
        observation_mode=observation_mode,
        num_simulations=num_simulations,
        end_to_end_active_decisions_per_sec_median=(
            end_to_end_active_decisions_per_sec_median
        ),
        closed_compact_loop_active_roots_per_sec=(
            closed_compact_loop_active_roots_per_sec
        ),
        compact_search_contract=compact_search_contract,
        compact_replay_index_contract=compact_replay_index_contract,
    )

    result = {
        "ok": not problems,
        "problems": problems,
        "compact_search_service_profile": compact_search_service_profile,
        "measurement_claim": (
            "resident_search_plus_fresh_h2d_and_policy_readback; "
            "host observation setup and replay edge are separate timing fields"
        ),
        "primary_gate_metric": "end_to_end_active_decisions_per_sec_median",
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
            "compact_visual_observation_source": compact_visual_observation_source,
            "visual_encoder": visual_encoder,
            "visual_channels": config.get("visual_channels"),
            "legal_mask_profile": config.get("legal_mask_profile"),
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
            "closed_loop_steps": closed_loop_steps,
            "closed_loop_action_only_profile": bool(
                config.get("closed_loop_action_only_profile", False)
            ),
            "closed_loop_deferred_payload_profile": bool(
                config.get("closed_loop_deferred_payload_profile", False)
            ),
            "closed_loop_overlap_payload_profile": bool(
                config.get("closed_loop_overlap_payload_profile", False)
            ),
            "closed_loop_replay_index": closed_loop_replay_index_enabled,
            "native_actor_buffer": bool(config.get("native_actor_buffer", False)),
            "policy_kind": "gumbel_muzero_policy",
            "compute": config.get("compute"),
            "hybrid_refresh_observation_stack": bool(
                config.get("hybrid_refresh_observation_stack", True)
            ),
            "hybrid_borrow_single_actor_render_state": bool(
                config.get("hybrid_borrow_single_actor_render_state", False)
            ),
            "hybrid_persistent_compact_render_state_buffer": bool(
                config.get("hybrid_persistent_compact_render_state_buffer", False)
            ),
            "compact_visual_resident_sync": bool(
                config.get("compact_visual_resident_sync", True)
            ),
            "persistent_renderer_async_device_only_profile": bool(
                config.get("persistent_renderer_async_device_only_profile", False)
            ),
            "persistent_vectorized_delta_pack_profile": bool(
                config.get("persistent_vectorized_delta_pack_profile", False)
            ),
            "compact_root_copy_observation": bool(
                config.get("compact_root_copy_observation", True)
            ),
        },
        "counts": counts,
        "observation": observation,
        "timing": {
            "host_observation_setup_sec": host_observation_setup_sec,
            "host_setup_plus_fresh_boundary_sec": host_setup_plus_fresh_boundary_sec,
            "host_setup_plus_fresh_boundary_active_decisions_per_sec": (
                host_setup_plus_fresh_boundary_active_decisions_per_sec
            ),
            "compact_replay_index_timing_sec": compact_replay_index_timing_sec,
            "closed_one_step_search_replay_edge_sec": (
                closed_one_step_search_replay_edge_sec
            ),
            "closed_one_step_search_replay_edge_active_decisions_per_sec": (
                closed_one_step_search_replay_edge_active_decisions_per_sec
            ),
            "closed_compact_loop_active_roots_per_sec": (
                closed_compact_loop_active_roots_per_sec
            ),
            "closed_compact_loop_slowest_bucket": (
                None
                if closed_compact_loop_contract is None
                else closed_compact_loop_contract["slowest_bucket"]
            ),
            "closed_compact_loop_bucket_totals_sec": (
                None
                if closed_compact_loop_contract is None
                else closed_compact_loop_contract["bucket_totals_sec"]
            ),
            "closed_compact_loop_next_step_bucket_totals_sec": (
                None
                if closed_compact_loop_contract is None
                else closed_compact_loop_contract["next_step_bucket_totals_sec"]
            ),
            "closed_compact_loop_bucket_fraction_of_total": (
                None
                if closed_compact_loop_contract is None
                else closed_compact_loop_contract["bucket_fraction_of_total"]
            ),
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
            "steady_search_plus_h2d_plus_policy_d2h_median_sec": (
                steady_search_plus_h2d_plus_policy_d2h_median_sec
            ),
            "steady_search_plus_h2d_plus_policy_d2h_roots_per_sec": (
                steady_search_plus_h2d_plus_policy_d2h_roots_per_sec
            ),
            "end_to_end_steady_times_sec": end_to_end_steady_times_sec,
            "end_to_end_steady_median_sec": end_to_end_steady_median_sec,
            "end_to_end_h2d_times_sec": end_to_end_h2d_times_sec,
            "end_to_end_h2d_median_sec": end_to_end_h2d_median_sec,
            "end_to_end_search_times_sec": end_to_end_search_times_sec,
            "end_to_end_search_median_sec": end_to_end_search_median_sec,
            "end_to_end_d2h_times_sec": end_to_end_d2h_times_sec,
            "end_to_end_d2h_median_sec": end_to_end_d2h_median_sec,
            "env_rows_per_sec_median": env_rows_per_sec_median,
            "decisions_per_sec_median": decisions_per_sec_median,
            "active_decisions_per_sec_median": active_decisions_per_sec_median,
            "simulations_per_sec_median": simulations_per_sec_median,
            "end_to_end_decisions_per_sec_median": end_to_end_decisions_per_sec_median,
            "end_to_end_active_decisions_per_sec_median": (
                end_to_end_active_decisions_per_sec_median
            ),
            "end_to_end_simulations_per_sec_median": (
                end_to_end_simulations_per_sec_median
            ),
        },
        "output": {
            "actions": actions[:action_sample_count].astype(int).tolist(),
            "actions_sample_count": action_sample_count,
            "actions_total_count": int(actions.shape[0]),
            "actions_truncated": bool(action_sample_count < actions.shape[0]),
            **legality_summary,
            "action_histogram": np.bincount(actions, minlength=ACTION_COUNT)
            .astype(int)
            .tolist(),
            "action_weights_finite": finite_weights,
            "action_weights_normalized": normalized_weights,
            "action_weight_row_sum_min": float(active_row_sums.min()),
            "action_weight_row_sum_max": float(active_row_sums.max()),
            "action_weight_sample": action_weights[: min(root_batch_size, 4)]
            .astype(float)
            .tolist(),
            "compact_search_result_sample": {
                "root_index": active_indices[:compact_sample_count].astype(int).tolist(),
                "selected_action": actions[active_indices[:compact_sample_count]]
                .astype(int)
                .tolist(),
                "visit_policy": action_weights[active_indices[:compact_sample_count]]
                .astype(float)
                .tolist(),
                "root_value": compact_root_values_sample,
                "root_value_source": root_value_source,
            },
            "compact_search_contract": compact_search_contract,
            "compact_replay_index_contract": compact_replay_index_contract,
            "closed_compact_loop_contract": closed_compact_loop_contract,
        },
    }
    printed_result = (
        result
        if bool(config.get("emit_full_json", True))
        else _compact_benchmark_summary(result)
    )
    print(json.dumps(printed_result, indent=2, sort_keys=True))
    return result


@app.function(image=gpu_image, gpu=["L4", "T4"], timeout=10 * 60)
def mctx_synthetic_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    return _run_benchmark(config)


@app.function(image=gpu_image, gpu="H100", timeout=30 * 60)
def mctx_synthetic_benchmark_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _run_benchmark(config)


@app.local_entrypoint()
def main(
    batch_size: int = 16,
    num_simulations: int = 8,
    hidden_dim: int = 32,
    visual_channels: int = 8,
    obs_dim: int = 0,
    observation_mode: str = OBSERVATION_MODE_FLAT,
    compute: str = "l4",
    legal_mask_profile: str = "all3",
    observation_renderer_backend: str = "jax_gpu_persistent_policy_framebuffer_profile",
    compact_visual_observation_source: str = COMPACT_VISUAL_OBSERVATION_SOURCE_HOST,
    player_count: int = 2,
    body_capacity: int = 4,
    step_index: int = 0,
    fixture_paths: str = "",
    group_id: str = "",
    event_mode: str = "",
    rollout_steps: int = 2,
    seed: int = 123,
    decision_ms: float = 1000.0 / 60.0,
    actor_count: int = 0,
    actor_simulations: int = 4,
    actor_seed: int = 0,
    allow_unverified: bool = False,
    max_depth: int = 8,
    warmup_runs: int = 1,
    steady_runs: int = 5,
    closed_loop_steps: int = 0,
    closed_loop_action_only_profile: bool = False,
    closed_loop_deferred_payload_profile: bool = False,
    closed_loop_overlap_payload_profile: bool = False,
    closed_loop_replay_index: bool = True,
    native_actor_buffer: bool = False,
    emit_full_json: bool = True,
    hybrid_refresh_observation_stack: bool = True,
    hybrid_borrow_single_actor_render_state: bool = False,
    hybrid_persistent_compact_render_state_buffer: bool = False,
    compact_visual_resident_sync: bool = True,
    persistent_renderer_async_device_only_profile: bool = False,
    persistent_vectorized_delta_pack_profile: bool = False,
    compact_root_copy_observation: bool = True,
) -> None:
    config = {
        "batch_size": batch_size,
        "num_simulations": num_simulations,
        "hidden_dim": hidden_dim,
        "visual_channels": visual_channels,
        "obs_dim": obs_dim,
        "observation_mode": observation_mode,
        "compute": compute,
        "legal_mask_profile": legal_mask_profile,
        "observation_renderer_backend": observation_renderer_backend,
        "compact_visual_observation_source": compact_visual_observation_source,
        "player_count": player_count,
        "body_capacity": body_capacity,
        "step_index": step_index,
        "fixture_paths": fixture_paths,
        "group_id": group_id,
        "rollout_steps": rollout_steps,
        "seed": seed,
        "decision_ms": decision_ms,
        "actor_count": actor_count,
        "actor_simulations": actor_simulations,
        "actor_seed": actor_seed,
        "allow_unverified": allow_unverified,
        "max_depth": max_depth,
        "warmup_runs": warmup_runs,
        "steady_runs": steady_runs,
        "closed_loop_steps": closed_loop_steps,
        "closed_loop_action_only_profile": closed_loop_action_only_profile,
        "closed_loop_deferred_payload_profile": closed_loop_deferred_payload_profile,
        "closed_loop_overlap_payload_profile": closed_loop_overlap_payload_profile,
        "closed_loop_replay_index": closed_loop_replay_index,
        "native_actor_buffer": native_actor_buffer,
        "emit_full_json": emit_full_json,
        "hybrid_refresh_observation_stack": hybrid_refresh_observation_stack,
        "hybrid_borrow_single_actor_render_state": hybrid_borrow_single_actor_render_state,
        "hybrid_persistent_compact_render_state_buffer": (
            hybrid_persistent_compact_render_state_buffer
        ),
        "compact_visual_resident_sync": compact_visual_resident_sync,
        "persistent_renderer_async_device_only_profile": (
            persistent_renderer_async_device_only_profile
        ),
        "persistent_vectorized_delta_pack_profile": persistent_vectorized_delta_pack_profile,
        "compact_root_copy_observation": compact_root_copy_observation,
    }
    if event_mode:
        config["event_mode"] = event_mode
    normalized_compute = compute.strip().lower()
    if normalized_compute in {"h100", "gpu-h100"}:
        result = mctx_synthetic_benchmark_h100.remote(config)
    elif normalized_compute in {"l4", "t4", "gpu-l4-t4", "gpu-l4"}:
        result = mctx_synthetic_benchmark.remote(config)
    else:
        raise ValueError("compute must be one of l4, t4, gpu-l4-t4, h100, gpu-h100")
    printed_result = result if emit_full_json else _compact_benchmark_summary(result)
    print(json.dumps(printed_result, indent=2, sort_keys=True))
