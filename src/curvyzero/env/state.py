"""State containers for the deterministic simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EnvState:
    """Mutable simulator state.

    Arrays are fixed-shape so the reference implementation can later grow a
    vectorized or Numba-accelerated backend without changing semantics.
    """

    tick: int
    positions: np.ndarray
    headings: np.ndarray
    alive: np.ndarray
    death_tick: np.ndarray
    occupancy: np.ndarray
    rng: np.random.Generator

