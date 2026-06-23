"""Device-resident observation contract for compact search paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RESIDENT_OBSERVATION_BATCH_SCHEMA_ID = "curvyzero_resident_observation_batch/v1"
RESIDENT_OBSERVATION_OWNER = "ResidentObservationBatchV1"
COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1 = "host_array_v1"
COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1 = "resident_device_v1"


@dataclass(frozen=True, slots=True)
class ResidentObservationBatchV1:
    """Fresh device-owned observation stack for compact search consumers.

    This is a fail-closed contract. A consumer that asks for resident observations
    must use this handle and must not silently read host observations.
    """

    device_observation: Any
    root_device_observation: Any | None
    generation_id: int
    batch_size: int
    player_count: int
    stack_shape: tuple[int, int, int]
    dtype: str
    device: str
    row_major_order: bool
    fresh_for_step_index: int
    source_backend: str
    host_fallback_allowed: bool = False
    metadata: dict[str, Any] | None = None
    final_device_observation: Any | None = None
    root_final_device_observation: Any | None = None
    final_observation_row_mask: Any | None = None
    final_device_observation_rows: Any | None = None
    final_device_observation_row_indices: Any | None = None


__all__ = [
    "COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1",
    "COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1",
    "RESIDENT_OBSERVATION_BATCH_SCHEMA_ID",
    "RESIDENT_OBSERVATION_OWNER",
    "ResidentObservationBatchV1",
]
