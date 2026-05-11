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
    history = np.empty((seeds.shape[0], length), dtype=np.float64)
    for row, seed in enumerate(seeds):
        state = int(seed) & _UINT64_MASK
        for index in range(length):
            state = (state + _SPLITMIX64_INCREMENT) & _UINT64_MASK
            bits = _splitmix64(state)
            history[row, index] = ((bits >> 11) & ((1 << 53) - 1)) * _FLOAT64_UNIT
    return history


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


__all__ = [
    "RANDOM_TAPE_SOURCE_SEED_GENERATED",
    "SOURCE_RANDOM_HISTORY_IMPL_ID",
    "VectorSourceRandomError",
    "seeded_source_math_random_history",
]
