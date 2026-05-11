"""Masked vector-row reset helpers.

This module is a small production-facing reset boundary. It assumes callers
already prepared reset-template rows and chose reset seeds. It does not run
natural spawn, schedule lifecycle timers, generate seeds, or decide autoreset
policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


RESET_INFO_SCHEMA_ID = "curvyzero_vector_reset_info/v1"
TERMINAL_TRANSITION_SNAPSHOT_SCHEMA_ID = (
    "curvyzero_vector_terminal_transition_snapshot/v1"
)

TERMINAL_REASON_NONE = 0
TERMINAL_REASON_SURVIVOR_WIN = 1
TERMINAL_REASON_ALL_DEAD_DRAW = 2
TERMINAL_REASON_TIMEOUT_TRUNCATED = 3
TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED = 4
TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED = 5

RESET_SOURCE_MANUAL = 0
RESET_SOURCE_AUTORESET = 1
RESET_SOURCE_FIXTURE = 2
RESET_SOURCE_REPLAY = 3

EVENT_NONE = 0

TERMINAL_REASON_CODE_NAMES = {
    TERMINAL_REASON_NONE: "none",
    TERMINAL_REASON_SURVIVOR_WIN: "survivor_win",
    TERMINAL_REASON_ALL_DEAD_DRAW: "all_dead_draw",
    TERMINAL_REASON_TIMEOUT_TRUNCATED: "timeout_truncated",
    TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED: "event_overflow_truncated",
    TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED: "body_overflow_truncated",
}
RESET_SOURCE_CODE_NAMES = {
    RESET_SOURCE_MANUAL: "manual",
    RESET_SOURCE_AUTORESET: "autoreset",
    RESET_SOURCE_FIXTURE: "fixture",
    RESET_SOURCE_REPLAY: "replay",
}

_RESET_ROW_ARRAYS: tuple[tuple[str, Any], ...] = (
    ("episode_id", np.int64),
    ("episode_step", np.int32),
    ("env_active", bool),
    ("reset_pending", bool),
    ("done", bool),
    ("terminated", bool),
    ("truncated", bool),
    ("terminal_reason", np.int16),
    ("reset_seed", np.uint64),
    ("reset_source", np.int16),
)

_EVENT_ARRAY_NAMES = (
    "event_count",
    "event_mask",
    "event_type",
    "event_player",
    "event_other",
    "event_bool",
    "event_value_i",
    "event_value_f",
    "event_overflow",
    "event_overflow_attempts",
)


class VectorResetError(ValueError):
    """Raised when a vector reset input cannot satisfy the row-reset contract."""


def reset_arrays(
    target: Mapping[str, np.ndarray],
    reset_template: Mapping[str, np.ndarray],
    reset_mask: np.ndarray,
    *,
    reset_seed: np.ndarray | int,
    reset_source: np.ndarray | int = RESET_SOURCE_MANUAL,
    snapshot_array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Reset selected rows from a prepared template and return reset metadata.

    Mutation order is part of the contract: validate inputs, snapshot selected
    terminal rows from ``target``, copy selected rows from ``reset_template``,
    then stamp the next-episode lifecycle fields on selected rows.
    """

    mask = _bool_row_mask(reset_mask, "reset_mask")
    batch_size = mask.shape[0]

    _validate_reset_lifecycle_state(target, batch_size=batch_size, state_name="target")
    _validate_reset_lifecycle_state(
        reset_template,
        batch_size=batch_size,
        state_name="reset_template",
    )
    _validate_matching_state_arrays(target, reset_template, batch_size=batch_size)
    _validate_optional_clock_rows(target, batch_size=batch_size)
    _validate_optional_event_rows(target, batch_size=batch_size)
    _validate_optional_timer_rows(target, batch_size=batch_size)

    previous_episode_id = np.asarray(target["episode_id"]).copy()
    if bool((previous_episode_id[mask] == np.iinfo(np.int64).max).any()):
        raise VectorResetError("episode_id values cannot be incremented without overflow")

    reset_episode_id = previous_episode_id.copy()
    reset_episode_id[mask] += 1
    reset_seed_array = _row_reset_metadata_input_array(
        reset_seed,
        "reset_seed",
        dtype=np.uint64,
        mask=mask,
        current=target["reset_seed"],
    )
    reset_source_array = _row_reset_metadata_input_array(
        reset_source,
        "reset_source",
        dtype=np.int16,
        mask=mask,
        current=target["reset_source"],
    )
    if not bool(np.isin(reset_source_array[mask], tuple(RESET_SOURCE_CODE_NAMES)).all()):
        raise VectorResetError("reset_source values must be known reset source codes")

    snapshot = terminal_transition_snapshot(
        target,
        mask,
        array_names=snapshot_array_names,
    )

    for name, source_array in reset_template.items():
        target[name][mask, ...] = source_array[mask, ...]

    target["episode_id"][mask] = reset_episode_id[mask]
    target["episode_step"][mask] = 0
    target["env_active"][mask] = True
    target["reset_pending"][mask] = False
    target["done"][mask] = False
    target["terminated"][mask] = False
    target["truncated"][mask] = False
    target["terminal_reason"][mask] = TERMINAL_REASON_NONE
    target["reset_seed"][mask] = reset_seed_array[mask]
    target["reset_source"][mask] = reset_source_array[mask]
    _clear_optional_clock_rows(target, mask)
    _clear_optional_event_rows(target, mask)
    _clear_optional_timer_rows(target, mask)

    return {
        "schema": RESET_INFO_SCHEMA_ID,
        "reset_schema_id": RESET_INFO_SCHEMA_ID,
        "reset_count": int(mask.sum()),
        "reset_mask": mask.copy(),
        "reset_rows": np.flatnonzero(mask).astype(np.int32),
        "reset_episode_id": np.asarray(target["episode_id"]).copy(),
        "reset_seed": np.asarray(target["reset_seed"]).copy(),
        "reset_source": np.asarray(target["reset_source"]).copy(),
        "reset_episode_step": np.asarray(target["episode_step"]).copy(),
        "reset_env_active": np.asarray(target["env_active"]).copy(),
        "reset_pending": np.asarray(target["reset_pending"]).copy(),
        "reset_done": np.asarray(target["done"]).copy(),
        "reset_terminated": np.asarray(target["terminated"]).copy(),
        "reset_truncated": np.asarray(target["truncated"]).copy(),
        "reset_terminal_reason": np.asarray(target["terminal_reason"]).copy(),
        "terminal_transition_snapshot": snapshot,
    }


def terminal_transition_snapshot(
    state: Mapping[str, np.ndarray],
    final_mask: np.ndarray,
    *,
    array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Copy selected final transition rows before backing state mutation."""

    mask = _bool_row_mask(final_mask, "final_mask")
    batch_size = mask.shape[0]
    names = _snapshot_array_names(state, array_names)
    for name in names:
        _validate_leading_batch_dimension(state, name, batch_size, "terminal snapshot")

    return {
        "schema": TERMINAL_TRANSITION_SNAPSHOT_SCHEMA_ID,
        "final_mask": mask.copy(),
        "final_rows": np.flatnonzero(mask).astype(np.int32),
        "arrays": {
            name: np.asarray(state[name])[mask, ...].copy()
            for name in names
        },
    }


def _bool_row_mask(value: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != bool or mask.ndim != 1:
        raise VectorResetError(f"{name} must be a bool array with shape [B]")
    return mask


def _row_array(value: np.ndarray, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value)
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype or array.ndim != 1:
        raise VectorResetError(
            f"{name} must be {_dtype_phrase(expected_dtype)} array with shape [B]"
        )
    return array


def _validate_reset_lifecycle_state(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    state_name: str,
) -> None:
    for name, dtype in _RESET_ROW_ARRAYS:
        if name not in state:
            raise VectorResetError(f"{state_name} state is missing {name!r} for reset")
        array = _bool_row_mask(state[name], name) if dtype is bool else _row_array(
            state[name],
            name,
            dtype,
        )
        if array.shape != (batch_size,):
            raise VectorResetError(
                f"{state_name} state {name!r} must match reset_mask shape [B]"
            )

    episode_id = np.asarray(state["episode_id"])
    if bool((episode_id < 0).any()):
        raise VectorResetError(f"{state_name} episode_id values must be non-negative")
    episode_step = np.asarray(state["episode_step"])
    if bool((episode_step < 0).any()):
        raise VectorResetError(f"{state_name} episode_step values must be non-negative")

    terminal_reason = np.asarray(state["terminal_reason"])
    if not bool(np.isin(terminal_reason, tuple(TERMINAL_REASON_CODE_NAMES)).all()):
        raise VectorResetError(
            f"{state_name} terminal_reason values must be known terminal reason codes"
        )
    state_reset_source = np.asarray(state["reset_source"])
    if not bool(np.isin(state_reset_source, tuple(RESET_SOURCE_CODE_NAMES)).all()):
        raise VectorResetError(
            f"{state_name} reset_source values must be known reset source codes"
        )


def _validate_matching_state_arrays(
    target: Mapping[str, np.ndarray],
    reset_template: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> None:
    if target.keys() != reset_template.keys():
        raise VectorResetError("target and reset_template state keys must match")

    for name, target_value in target.items():
        target_array = np.asarray(target_value)
        template_array = np.asarray(reset_template[name])
        if target_array.shape != template_array.shape:
            raise VectorResetError(
                f"state array {name!r} shape differs during row reset"
            )
        if target_array.dtype != template_array.dtype:
            raise VectorResetError(
                f"state array {name!r} dtype differs during row reset"
            )
        if target_array.ndim < 1 or target_array.shape[0] != batch_size:
            raise VectorResetError(
                f"state array {name!r} must have leading reset_mask dimension"
            )


def _validate_optional_clock_rows(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> None:
    if "tick" in state:
        _validate_row_shape(_row_array(state["tick"], "tick", np.int32), "tick", batch_size)
    if "elapsed_ms" in state:
        _validate_row_shape(
            _row_array(state["elapsed_ms"], "elapsed_ms", np.float64),
            "elapsed_ms",
            batch_size,
        )


def _validate_optional_event_rows(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> None:
    if not any(name in state for name in _EVENT_ARRAY_NAMES):
        return
    for name in _EVENT_ARRAY_NAMES:
        if name not in state:
            raise VectorResetError(f"state is missing {name!r} for event row reset")

    event_count = _row_array(state["event_count"], "event_count", np.int16)
    _validate_row_shape(event_count, "event_count", batch_size)
    event_mask = _leading_array(state["event_mask"], "event_mask", np.bool_, batch_size)
    event_type = _leading_array(state["event_type"], "event_type", np.int16, batch_size)
    event_player = _leading_array(
        state["event_player"],
        "event_player",
        np.int16,
        batch_size,
    )
    event_other = _leading_array(state["event_other"], "event_other", np.int16, batch_size)
    event_bool = _leading_array(state["event_bool"], "event_bool", np.int8, batch_size)
    event_value_i = _leading_array(
        state["event_value_i"],
        "event_value_i",
        np.int32,
        batch_size,
    )
    event_value_f = _leading_array(
        state["event_value_f"],
        "event_value_f",
        np.float64,
        batch_size,
    )
    event_overflow = _bool_row_mask(state["event_overflow"], "event_overflow")
    _validate_row_shape(event_overflow, "event_overflow", batch_size)
    event_overflow_attempts = _row_array(
        state["event_overflow_attempts"],
        "event_overflow_attempts",
        np.int32,
    )
    _validate_row_shape(
        event_overflow_attempts,
        "event_overflow_attempts",
        batch_size,
    )

    if event_mask.ndim != 2:
        raise VectorResetError("event_mask must have shape [B,L]")
    for name, array in (
        ("event_type", event_type),
        ("event_player", event_player),
        ("event_other", event_other),
        ("event_bool", event_bool),
    ):
        if array.shape != event_mask.shape:
            raise VectorResetError(f"{name} must match event_mask shape [B,L]")
    expected_value_shape = (*event_mask.shape, 2)
    if event_value_i.shape != expected_value_shape:
        raise VectorResetError("event_value_i must have shape [B,L,2]")
    if event_value_f.shape != expected_value_shape:
        raise VectorResetError("event_value_f must have shape [B,L,2]")


def _validate_optional_timer_rows(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> None:
    if "timer_fired_count" not in state:
        return
    timer_fired_count = _row_array(
        state["timer_fired_count"],
        "timer_fired_count",
        np.int16,
    )
    _validate_row_shape(timer_fired_count, "timer_fired_count", batch_size)


def _row_reset_metadata_input_array(
    value: np.ndarray | int,
    name: str,
    *,
    dtype: Any,
    mask: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    current_array = _row_array(current, name, dtype)
    if current_array.shape != mask.shape:
        raise VectorResetError(f"{name} current value must match reset_mask shape [B]")

    result = current_array.copy()
    array = np.asarray(value)
    if array.ndim == 0:
        result[mask] = _metadata_scalar_value(value, name, dtype)
        return result
    if array.dtype != np.dtype(dtype) or array.ndim != 1:
        raise VectorResetError(
            f"{name} must be {_dtype_phrase(np.dtype(dtype))} array with shape [B]"
        )
    if array.shape != mask.shape:
        raise VectorResetError(f"{name} and reset_mask must have matching shape [B]")
    result[mask] = array[mask]
    return result


def _metadata_scalar_value(value: np.ndarray | int, name: str, dtype: Any) -> Any:
    scalar_array = np.asarray(value)
    if scalar_array.dtype == np.dtype(bool) or scalar_array.dtype.kind not in ("i", "u"):
        raise VectorResetError(f"{name} scalar must be an integer")
    scalar = int(scalar_array.item())
    dtype_info = np.iinfo(np.dtype(dtype))
    if scalar < 0 and np.dtype(dtype).kind == "u":
        raise VectorResetError(f"{name} scalar must be non-negative")
    if scalar < dtype_info.min or scalar > dtype_info.max:
        raise VectorResetError(f"{name} scalar must fit in {np.dtype(dtype).name}")
    return np.asarray(scalar, dtype=dtype).item()


def _snapshot_array_names(
    state: Mapping[str, np.ndarray],
    array_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if array_names is None:
        return tuple(state.keys())
    if isinstance(array_names, str):
        raise VectorResetError("array_names must be a sequence of state array names")

    names = tuple(array_names)
    for name in names:
        if not isinstance(name, str):
            raise VectorResetError("array_names must contain only strings")
        if name not in state:
            raise VectorResetError(f"state array {name!r} is missing for terminal snapshot")
    return names


def _leading_array(
    value: np.ndarray,
    name: str,
    dtype: Any,
    batch_size: int,
) -> np.ndarray:
    array = np.asarray(value)
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype or array.ndim < 1 or array.shape[0] != batch_size:
        raise VectorResetError(
            f"{name} must be {_dtype_phrase(expected_dtype)} array with leading shape [B]"
        )
    return array


def _validate_leading_batch_dimension(
    state: Mapping[str, np.ndarray],
    name: str,
    batch_size: int,
    context: str,
) -> None:
    array = np.asarray(state[name])
    if array.ndim < 1 or array.shape[0] != batch_size:
        raise VectorResetError(
            f"state array {name!r} must have leading {context} mask dimension"
        )


def _validate_row_shape(array: np.ndarray, name: str, batch_size: int) -> None:
    if array.shape != (batch_size,):
        raise VectorResetError(f"{name} must have shape [B]")


def _dtype_phrase(dtype: np.dtype) -> str:
    article = "an" if dtype.name.startswith("int") else "a"
    return f"{article} {dtype.name}"


def _clear_optional_clock_rows(state: Mapping[str, np.ndarray], mask: np.ndarray) -> None:
    if "tick" in state:
        state["tick"][mask] = 0
    if "elapsed_ms" in state:
        state["elapsed_ms"][mask] = 0.0


def _clear_optional_event_rows(state: Mapping[str, np.ndarray], mask: np.ndarray) -> None:
    if not any(name in state for name in _EVENT_ARRAY_NAMES):
        return
    state["event_count"][mask] = 0
    state["event_mask"][mask, ...] = False
    state["event_type"][mask, ...] = EVENT_NONE
    state["event_player"][mask, ...] = -1
    state["event_other"][mask, ...] = -1
    state["event_bool"][mask, ...] = -1
    state["event_value_i"][mask, ...] = 0
    state["event_value_f"][mask, ...] = 0.0
    state["event_overflow"][mask] = False
    state["event_overflow_attempts"][mask] = 0


def _clear_optional_timer_rows(state: Mapping[str, np.ndarray], mask: np.ndarray) -> None:
    if "timer_fired_count" in state:
        state["timer_fired_count"][mask] = 0


__all__ = [
    "EVENT_NONE",
    "RESET_INFO_SCHEMA_ID",
    "RESET_SOURCE_AUTORESET",
    "RESET_SOURCE_CODE_NAMES",
    "RESET_SOURCE_FIXTURE",
    "RESET_SOURCE_MANUAL",
    "RESET_SOURCE_REPLAY",
    "TERMINAL_REASON_ALL_DEAD_DRAW",
    "TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED",
    "TERMINAL_REASON_CODE_NAMES",
    "TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED",
    "TERMINAL_REASON_NONE",
    "TERMINAL_REASON_SURVIVOR_WIN",
    "TERMINAL_REASON_TIMEOUT_TRUNCATED",
    "TERMINAL_TRANSITION_SNAPSHOT_SCHEMA_ID",
    "VectorResetError",
    "reset_arrays",
    "terminal_transition_snapshot",
]
