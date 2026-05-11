"""Public vector autoreset planning and mutation contracts.

This module does not step environments. The planner is a small handoff helper
for callers that need to prove terminal transition data was staged before
asking ``vector_reset.reset_arrays`` to reset rows; the apply helper performs
that narrow mutation after staging.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import vector_reset


AUTORESET_PLAN_SCHEMA_ID = "curvyzero_vector_autoreset_plan/v1"
AUTORESET_APPLY_SCHEMA_ID = "curvyzero_vector_autoreset_apply/v1"
AUTORESET_TERMINAL_SNAPSHOT_SCHEMA_ID = (
    "curvyzero_vector_autoreset_terminal_snapshot/v1"
)
AUTORESET_SURFACE = "public_reset_autoreset_ordering_1v1_no_bonus_v0"
RESET_ARRAYS_FUNCTION = "curvyzero.env.vector_reset.reset_arrays"

_REQUIRED_LIFECYCLE_ARRAYS: tuple[tuple[str, Any], ...] = (
    ("done", bool),
    ("terminated", bool),
    ("truncated", bool),
    ("episode_id", np.int64),
    ("episode_step", np.int32),
)


class VectorAutoresetError(ValueError):
    """Raised when a public autoreset handoff cannot satisfy the contract."""


def plan_autoreset_rows(
    lifecycle: Mapping[str, np.ndarray],
    *,
    final_observation: np.ndarray | None,
    final_reward_map: np.ndarray | None,
    reset_seed: np.ndarray | None,
    reset_source: np.ndarray | None,
    autoreset_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate and stage public autoreset metadata without mutating state.

    Without an explicit ``autoreset_mask``, only rows whose ``done`` flag is true
    are selected. If a caller passes ``autoreset_mask``, that mask is treated as
    an explicit override and any selected non-done rows are reported.
    """

    done = _required_bool_lifecycle_array(lifecycle, "done")
    batch_size = done.shape[0]
    terminated = _required_bool_lifecycle_array(
        lifecycle,
        "terminated",
        batch_size=batch_size,
    )
    truncated = _required_bool_lifecycle_array(
        lifecycle,
        "truncated",
        batch_size=batch_size,
    )
    episode_id = _required_row_lifecycle_array(
        lifecycle,
        "episode_id",
        np.int64,
        batch_size=batch_size,
    )
    episode_step = _required_row_lifecycle_array(
        lifecycle,
        "episode_step",
        np.int32,
        batch_size=batch_size,
    )
    _validate_done_semantics(done, terminated, truncated)

    mask_source = "caller" if autoreset_mask is not None else "done"
    selected_mask = (
        _bool_row_mask(autoreset_mask, "autoreset_mask", batch_size=batch_size)
        if autoreset_mask is not None
        else done.copy()
    )
    selected_rows = np.flatnonzero(selected_mask).astype(np.int32)

    final_observation_array = _required_leading_array(
        final_observation,
        "final_observation",
        batch_size,
    )
    final_reward_map_array = _required_leading_array(
        final_reward_map,
        "final_reward_map",
        batch_size,
    )
    reset_seed_array = _required_reset_seed_array(reset_seed, batch_size=batch_size)
    reset_source_array = _required_reset_source_array(reset_source, batch_size=batch_size)

    _validate_selected_rows_present(
        final_observation_array,
        "final_observation",
        selected_mask,
    )
    _validate_selected_rows_present(
        final_reward_map_array,
        "final_reward_map",
        selected_mask,
    )
    _validate_episode_metadata(episode_id, episode_step, selected_mask)

    final_transition_snapshot = _final_transition_snapshot(
        selected_mask,
        selected_rows=selected_rows,
        done=done,
        terminated=terminated,
        truncated=truncated,
        episode_id=episode_id,
        episode_step=episode_step,
        final_observation=final_observation_array,
        final_reward_map=final_reward_map_array,
    )
    reset_episode_id = episode_id[selected_mask].copy() + np.asarray(1, dtype=np.int64)
    reset_episode_step = np.zeros(selected_rows.shape, dtype=np.int32)
    reset_metadata = {
        "rows": selected_rows.copy(),
        "final_episode_id": episode_id[selected_mask].copy(),
        "final_episode_step": episode_step[selected_mask].copy(),
        "reset_episode_id": reset_episode_id,
        "reset_episode_step": reset_episode_step,
        "reset_seed": reset_seed_array[selected_mask].copy(),
        "reset_source": reset_source_array[selected_mask].copy(),
    }
    selected_non_done_mask = selected_mask & ~done

    return {
        "schema": AUTORESET_PLAN_SCHEMA_ID,
        "surface": AUTORESET_SURFACE,
        "mutates_state": False,
        "reset_arrays_function": RESET_ARRAYS_FUNCTION,
        "mask_source": mask_source,
        "explicit_autoreset_mask": autoreset_mask is not None,
        "autoreset_count": int(selected_mask.sum()),
        "row_count": int(selected_mask.sum()),
        "autoreset_mask": selected_mask.copy(),
        "eligible_mask": done.copy(),
        "row_ids": selected_rows.copy(),
        "autoreset_rows": selected_rows.copy(),
        "selected_non_done_rows": np.flatnonzero(selected_non_done_mask).astype(np.int32),
        "done": done[selected_mask].copy(),
        "terminated": terminated[selected_mask].copy(),
        "truncated": truncated[selected_mask].copy(),
        "final_transition_snapshot": final_transition_snapshot,
        "reset_metadata": reset_metadata,
        "reset_arrays_kwargs": {
            "reset_mask": selected_mask.copy(),
            "reset_seed": reset_seed_array.copy(),
            "reset_source": reset_source_array.copy(),
        },
    }


def apply_autoreset_rows(
    target: Mapping[str, np.ndarray],
    reset_template: Mapping[str, np.ndarray],
    lifecycle: Mapping[str, np.ndarray],
    *,
    final_observation: np.ndarray | None,
    final_reward_map: np.ndarray | None,
    reset_seed: np.ndarray | None,
    reset_source: np.ndarray | None,
    autoreset_mask: np.ndarray | None = None,
    snapshot_array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Stage final rows, reset selected rows, and return both contracts."""

    plan = plan_autoreset_rows(
        lifecycle,
        final_observation=final_observation,
        final_reward_map=final_reward_map,
        reset_seed=reset_seed,
        reset_source=reset_source,
        autoreset_mask=autoreset_mask,
    )
    reset_kwargs = plan["reset_arrays_kwargs"]
    reset_info = vector_reset.reset_arrays(
        target,
        reset_template,
        plan["autoreset_mask"],
        reset_seed=reset_kwargs["reset_seed"],
        reset_source=reset_kwargs["reset_source"],
        snapshot_array_names=snapshot_array_names,
    )

    return {
        "schema": AUTORESET_APPLY_SCHEMA_ID,
        "surface": AUTORESET_SURFACE,
        "mutates_state": True,
        "reset_arrays_function": RESET_ARRAYS_FUNCTION,
        "plan": plan,
        "reset_info": reset_info,
        "final_transition_snapshot": plan["final_transition_snapshot"],
        "reset_metadata": plan["reset_metadata"],
        "autoreset_count": plan["autoreset_count"],
        "row_count": plan["row_count"],
        "autoreset_mask": plan["autoreset_mask"].copy(),
        "autoreset_rows": plan["autoreset_rows"].copy(),
        "selected_non_done_rows": plan["selected_non_done_rows"].copy(),
    }


def _final_transition_snapshot(
    selected_mask: np.ndarray,
    *,
    selected_rows: np.ndarray,
    done: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    episode_id: np.ndarray,
    episode_step: np.ndarray,
    final_observation: np.ndarray,
    final_reward_map: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema": AUTORESET_TERMINAL_SNAPSHOT_SCHEMA_ID,
        "captured_before_reset": True,
        "rows": selected_rows.copy(),
        "done": done[selected_mask].copy(),
        "terminated": terminated[selected_mask].copy(),
        "truncated": truncated[selected_mask].copy(),
        "final_episode_id": episode_id[selected_mask].copy(),
        "final_episode_step": episode_step[selected_mask].copy(),
        "final_observation": final_observation[selected_mask, ...].copy(),
        "final_reward_map": final_reward_map[selected_mask, ...].copy(),
    }


def _required_bool_lifecycle_array(
    lifecycle: Mapping[str, np.ndarray],
    name: str,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    if name not in lifecycle:
        raise VectorAutoresetError(f"lifecycle is missing {name!r}")
    return _bool_row_mask(lifecycle[name], name, batch_size=batch_size)


def _required_row_lifecycle_array(
    lifecycle: Mapping[str, np.ndarray],
    name: str,
    dtype: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    if name not in lifecycle:
        raise VectorAutoresetError(f"lifecycle is missing {name!r}")
    array = _row_array(lifecycle[name], name, dtype)
    if array.shape != (batch_size,):
        raise VectorAutoresetError(f"{name} must match done shape [B]")
    return array


def _required_leading_array(
    value: np.ndarray | None,
    name: str,
    batch_size: int,
) -> np.ndarray:
    if value is None:
        raise VectorAutoresetError(f"{name} is required for selected autoreset rows")
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[0] != batch_size:
        raise VectorAutoresetError(f"{name} must have leading shape [B]")
    return array


def _required_reset_seed_array(
    value: np.ndarray | None,
    *,
    batch_size: int,
) -> np.ndarray:
    if value is None:
        raise VectorAutoresetError("reset_seed is required for selected autoreset rows")
    array = _row_array(value, "reset_seed", np.uint64)
    if array.shape != (batch_size,):
        raise VectorAutoresetError("reset_seed must have shape [B]")
    return array


def _required_reset_source_array(
    value: np.ndarray | None,
    *,
    batch_size: int,
) -> np.ndarray:
    if value is None:
        raise VectorAutoresetError("reset_source is required for selected autoreset rows")
    array = _row_array(value, "reset_source", np.int16)
    if array.shape != (batch_size,):
        raise VectorAutoresetError("reset_source must have shape [B]")
    if not bool(np.isin(array, tuple(vector_reset.RESET_SOURCE_CODE_NAMES)).all()):
        raise VectorAutoresetError("reset_source values must be known reset source codes")
    return array


def _validate_done_semantics(
    done: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> None:
    if not np.array_equal(done, np.logical_or(terminated, truncated)):
        raise VectorAutoresetError("done must equal terminated OR truncated")


def _validate_selected_rows_present(
    array: np.ndarray,
    name: str,
    selected_mask: np.ndarray,
) -> None:
    if not bool(selected_mask.any()) or array.dtype != np.dtype(object):
        return
    selected = array[selected_mask, ...]
    if any(value is None for value in selected.ravel()):
        raise VectorAutoresetError(
            f"{name} contains missing metadata for selected autoreset rows"
        )


def _validate_episode_metadata(
    episode_id: np.ndarray,
    episode_step: np.ndarray,
    selected_mask: np.ndarray,
) -> None:
    if bool((episode_id < 0).any()):
        raise VectorAutoresetError("episode_id values must be non-negative")
    if bool((episode_step < 0).any()):
        raise VectorAutoresetError("episode_step values must be non-negative")
    if bool((episode_id[selected_mask] == np.iinfo(np.int64).max).any()):
        raise VectorAutoresetError("episode_id values cannot be incremented without overflow")


def _bool_row_mask(
    value: np.ndarray,
    name: str,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != bool or mask.ndim != 1:
        raise VectorAutoresetError(f"{name} must be a bool array with shape [B]")
    if batch_size is not None and mask.shape != (batch_size,):
        raise VectorAutoresetError(f"{name} must match done shape [B]")
    return mask


def _row_array(value: np.ndarray, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value)
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype or array.ndim != 1:
        raise VectorAutoresetError(
            f"{name} must be {_dtype_phrase(expected_dtype)} array with shape [B]"
        )
    return array


def _dtype_phrase(dtype: np.dtype) -> str:
    article = "an" if dtype.name.startswith("int") else "a"
    return f"{article} {dtype.name}"


__all__ = [
    "AUTORESET_APPLY_SCHEMA_ID",
    "AUTORESET_PLAN_SCHEMA_ID",
    "AUTORESET_SURFACE",
    "AUTORESET_TERMINAL_SNAPSHOT_SCHEMA_ID",
    "RESET_ARRAYS_FUNCTION",
    "VectorAutoresetError",
    "apply_autoreset_rows",
    "plan_autoreset_rows",
]
