"""Seeded source-shaped Math.random history for vector reset rows."""

from __future__ import annotations

from typing import Any

import numpy as np


RANDOM_TAPE_SOURCE_SEED_GENERATED = "seed_generated_source_random_history"
SOURCE_RANDOM_HISTORY_IMPL_ID = "curvyzero_seeded_source_math_random_history/v0"

_UINT64_MASK = (1 << 64) - 1
_SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX64_MUL_A = 0xBF58476D1CE4E5B9
_SPLITMIX64_MUL_B = 0x94D049BB133111EB
_UINT64_SHIFT_A = 30
_UINT64_SHIFT_B = 27
_UINT64_SHIFT_C = 31
_RANDOM_FLOAT_SHIFT = 11
_RANDOM_FLOAT_MASK = (1 << 53) - 1
_FLOAT64_UNIT = 1.0 / float(1 << 53)


class VectorSourceRandomError(ValueError):
    """Raised when source-shaped seeded random history inputs are invalid."""


def seeded_source_math_random_history(
    reset_seed: Any,
    *,
    length: int,
) -> np.ndarray:
    """Return row-local deterministic float64 Math.random histories.

    The generator is intentionally a CurvyZero-owned seeded history format, not
    a claim of V8 ``Math.random`` bit parity. Values are 53-bit ``float64`` in
    ``[0, 1)`` so the same history can be fed to ``CurvyTronSourceEnv`` or the
    vector row-local random tape path.
    """

    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise VectorSourceRandomError("length must be a positive integer")

    seeds = _seed_rows(reset_seed)
    increments = (
        np.arange(1, length + 1, dtype=np.uint64)
        * np.uint64(_SPLITMIX64_INCREMENT)
    )
    states = seeds[:, np.newaxis] + increments[np.newaxis, :]
    bits = _splitmix64_array(states)
    mantissa = (bits >> np.uint64(_RANDOM_FLOAT_SHIFT)) & np.uint64(_RANDOM_FLOAT_MASK)
    return mantissa.astype(np.float64, copy=False) * _FLOAT64_UNIT


def _seed_rows(reset_seed: Any) -> np.ndarray:
    try:
        seeds = np.asarray(reset_seed)
    except (TypeError, ValueError) as exc:
        raise VectorSourceRandomError("reset_seed must be an integer row array") from exc
    if seeds.ndim != 1:
        raise VectorSourceRandomError("reset_seed must have shape [B]")
    if not np.issubdtype(seeds.dtype, np.integer):
        raise VectorSourceRandomError("reset_seed must be integer")
    if bool((seeds < 0).any()):
        raise VectorSourceRandomError("reset_seed values must be non-negative")
    return seeds.astype(np.uint64, copy=False)


def _splitmix64(value: int) -> int:
    mixed = value & _UINT64_MASK
    mixed = (mixed ^ (mixed >> 30)) * _SPLITMIX64_MUL_A & _UINT64_MASK
    mixed = (mixed ^ (mixed >> 27)) * _SPLITMIX64_MUL_B & _UINT64_MASK
    return (mixed ^ (mixed >> 31)) & _UINT64_MASK


def _splitmix64_array(values: np.ndarray) -> np.ndarray:
    mixed = np.asarray(values, dtype=np.uint64)
    mixed = (mixed ^ (mixed >> np.uint64(_UINT64_SHIFT_A))) * np.uint64(
        _SPLITMIX64_MUL_A
    )
    mixed = (mixed ^ (mixed >> np.uint64(_UINT64_SHIFT_B))) * np.uint64(
        _SPLITMIX64_MUL_B
    )
    return mixed ^ (mixed >> np.uint64(_UINT64_SHIFT_C))


__all__ = [
    "RANDOM_TAPE_SOURCE_SEED_GENERATED",
    "SOURCE_RANDOM_HISTORY_IMPL_ID",
    "VectorSourceRandomError",
    "seeded_source_math_random_history",
]
