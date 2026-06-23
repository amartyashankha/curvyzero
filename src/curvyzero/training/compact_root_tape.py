"""Fixed-root tapes for confounder-controlled compact search comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
)
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactSearchResultV1
from curvyzero.training.compact_policy_row_bridge import (
    validate_compact_search_result_identity_v1,
)
from curvyzero.training.compact_search_service import CompactSearchServiceV1
from curvyzero.training.compact_search_service import (
    compact_search_comparison_telemetry_v1,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


COMPACT_ROOT_TAPE_SCHEMA_ID = "curvyzero_compact_root_tape/v1"
COMPACT_ROOT_TAPE_RECORD_SCHEMA_ID = "curvyzero_compact_root_tape_record/v1"
COMPACT_ROOT_TAPE_COMPARISON_SCHEMA_ID = "curvyzero_compact_root_tape_comparison/v1"


@dataclass(frozen=True, slots=True)
class CompactRootTapeRecordV1:
    """One root batch captured before a compact search backend runs."""

    schema_id: str
    record_index: int
    observation: np.ndarray
    legal_mask: np.ndarray
    active_root_mask: np.ndarray
    to_play: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    policy_env_id: np.ndarray
    target_reward: np.ndarray
    done_root: np.ndarray
    final_observation: np.ndarray | None
    final_observation_row_mask: np.ndarray
    terminal_row_mask: np.ndarray
    autoreset_row_mask: np.ndarray
    root_metadata: dict[str, Any]
    capture_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactRootTapeV1:
    """A deterministic root-batch tape for offline backend comparison."""

    schema_id: str
    records: tuple[CompactRootTapeRecordV1, ...]
    metadata: dict[str, Any]


class InMemoryCompactRootTapeRecorderV1:
    """Small recorder that can be plugged into ``CompactRolloutSlab``."""

    def __init__(
        self,
        *,
        tape_label: str,
        allow_resident_host_snapshot: bool = False,
        max_records: int | None = None,
    ) -> None:
        label = str(tape_label)
        if not label:
            raise ReplayCompatibilityError("tape_label must be non-empty")
        self.tape_label = label
        self.allow_resident_host_snapshot = bool(allow_resident_host_snapshot)
        self.max_records = None if max_records is None else _positive_int(max_records)
        self._records: list[CompactRootTapeRecordV1] = []
        self._skipped_record_count = 0

    @property
    def record_count(self) -> int:
        """Number of captured root batches."""

        return len(self._records)

    @property
    def skipped_record_count(self) -> int:
        """Number of root batches skipped after ``max_records`` was reached."""

        return int(self._skipped_record_count)

    def record_root_batch(
        self,
        root_batch: CompactRootBatchV1,
        *,
        record_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Capture a root batch as an immutable host snapshot."""

        if self.max_records is not None and len(self._records) >= self.max_records:
            self._skipped_record_count += 1
            return
        host_snapshot = None
        if (
            str(root_batch.observation_source)
            == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        ):
            raise ReplayCompatibilityError(
                "resident root-batch tape capture requires a real explicit "
                "device-to-host snapshot; the recorder must not use the "
                "non-authoritative host observation"
            )
        capture_metadata = {
            "tape_label": self.tape_label,
            "recorder_allow_resident_host_snapshot": self.allow_resident_host_snapshot,
        }
        if metadata:
            capture_metadata.update(_plain_metadata(metadata))
        self._records.append(
            compact_root_tape_record_v1_from_root_batch(
                root_batch,
                record_index=int(record_index),
                capture_metadata=capture_metadata,
                host_observation_snapshot=host_snapshot,
            )
        )

    def build_tape(self, *, metadata: Mapping[str, Any] | None = None) -> CompactRootTapeV1:
        """Return the captured tape."""

        tape_metadata = {
            "tape_label": self.tape_label,
            "record_count": len(self._records),
            "skipped_record_count": int(self._skipped_record_count),
            "max_records": self.max_records,
        }
        if metadata:
            tape_metadata.update(_plain_metadata(metadata))
        return CompactRootTapeV1(
            schema_id=COMPACT_ROOT_TAPE_SCHEMA_ID,
            records=tuple(self._records),
            metadata=tape_metadata,
        )


def compact_root_tape_record_v1_from_root_batch(
    root_batch: CompactRootBatchV1,
    *,
    record_index: int,
    capture_metadata: Mapping[str, Any] | None = None,
    host_observation_snapshot: np.ndarray | None = None,
) -> CompactRootTapeRecordV1:
    """Copy one root batch into a replayable host-owned tape record."""

    source = str(root_batch.observation_source)
    if source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1:
        if host_observation_snapshot is None:
            raise ReplayCompatibilityError(
                "resident root-batch tape capture requires host_observation_snapshot"
            )
        observation = np.asarray(host_observation_snapshot)
        if observation.shape != np.asarray(root_batch.observation).shape:
            raise ReplayCompatibilityError(
                "host_observation_snapshot shape must match root_batch.observation"
            )
        snapshot_metadata = {
            "root_tape_resident_source": True,
            "root_tape_observation_source": "explicit_host_snapshot",
            "root_tape_original_observation_source": source,
        }
    elif source == COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1:
        if host_observation_snapshot is not None:
            raise ReplayCompatibilityError(
                "host root-batch tape capture must not receive host_observation_snapshot"
            )
        observation = np.asarray(root_batch.observation)
        snapshot_metadata = {
            "root_tape_resident_source": False,
            "root_tape_observation_source": "host_array",
            "root_tape_original_observation_source": source,
        }
    else:
        raise ReplayCompatibilityError(f"unknown root batch observation_source {source!r}")

    root_metadata = _plain_metadata(root_batch.metadata)
    root_metadata.update(snapshot_metadata)
    root_metadata["observation_source"] = COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
    root_metadata["host_observation_authoritative"] = True
    root_metadata["resident_device_observation_authoritative"] = False

    return CompactRootTapeRecordV1(
        schema_id=COMPACT_ROOT_TAPE_RECORD_SCHEMA_ID,
        record_index=int(record_index),
        observation=np.asarray(observation).copy(),
        legal_mask=np.asarray(root_batch.legal_mask, dtype=np.bool_).copy(),
        active_root_mask=np.asarray(root_batch.active_root_mask, dtype=np.bool_).copy(),
        to_play=np.asarray(root_batch.to_play, dtype=np.int64).copy(),
        env_row=np.asarray(root_batch.env_row, dtype=np.int32).copy(),
        player=np.asarray(root_batch.player, dtype=np.int16).copy(),
        policy_env_id=np.asarray(root_batch.policy_env_id, dtype=np.int64).copy(),
        target_reward=np.asarray(root_batch.target_reward, dtype=np.float32).copy(),
        done_root=np.asarray(root_batch.done_root, dtype=np.bool_).copy(),
        final_observation=None
        if root_batch.final_observation is None
        else np.asarray(root_batch.final_observation).copy(),
        final_observation_row_mask=np.asarray(
            root_batch.final_observation_row_mask,
            dtype=np.bool_,
        ).copy(),
        terminal_row_mask=np.asarray(root_batch.terminal_row_mask, dtype=np.bool_).copy(),
        autoreset_row_mask=np.asarray(
            root_batch.autoreset_row_mask,
            dtype=np.bool_,
        ).copy(),
        root_metadata=root_metadata,
        capture_metadata=_plain_metadata(capture_metadata or {}),
    )


def compact_root_batch_v1_from_tape_record(
    record: CompactRootTapeRecordV1,
) -> CompactRootBatchV1:
    """Rehydrate a tape record as a host-owned compact root batch."""

    if record.schema_id != COMPACT_ROOT_TAPE_RECORD_SCHEMA_ID:
        raise ReplayCompatibilityError("unknown compact root tape record schema")
    metadata = dict(record.root_metadata)
    metadata.update(
        {
            "root_tape_replay": True,
            "root_tape_record_index": int(record.record_index),
            "observation_source": COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
            "host_observation_authoritative": True,
            "resident_device_observation_authoritative": False,
        }
    )
    return CompactRootBatchV1(
        observation=np.asarray(record.observation).copy(),
        legal_mask=np.asarray(record.legal_mask, dtype=np.bool_).copy(),
        active_root_mask=np.asarray(record.active_root_mask, dtype=np.bool_).copy(),
        to_play=np.asarray(record.to_play, dtype=np.int64).copy(),
        env_row=np.asarray(record.env_row, dtype=np.int32).copy(),
        player=np.asarray(record.player, dtype=np.int16).copy(),
        policy_env_id=np.asarray(record.policy_env_id, dtype=np.int64).copy(),
        target_reward=np.asarray(record.target_reward, dtype=np.float32).copy(),
        done_root=np.asarray(record.done_root, dtype=np.bool_).copy(),
        final_observation=None
        if record.final_observation is None
        else np.asarray(record.final_observation).copy(),
        final_observation_row_mask=np.asarray(
            record.final_observation_row_mask,
            dtype=np.bool_,
        ).copy(),
        terminal_row_mask=np.asarray(record.terminal_row_mask, dtype=np.bool_).copy(),
        autoreset_row_mask=np.asarray(record.autoreset_row_mask, dtype=np.bool_).copy(),
        metadata=metadata,
        resident_observation=None,
        observation_source=COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
    )


def run_compact_root_tape_comparison_v1(
    tape: CompactRootTapeV1,
    *,
    services: Mapping[str, CompactSearchServiceV1],
    reference_label: str,
) -> dict[str, Any]:
    """Run each service on the same saved roots and aggregate comparison metrics."""

    if tape.schema_id != COMPACT_ROOT_TAPE_SCHEMA_ID:
        raise ReplayCompatibilityError("unknown compact root tape schema")
    if not tape.records:
        raise ReplayCompatibilityError("compact root tape must contain at least one record")
    service_map = {str(label): service for label, service in services.items()}
    if len(service_map) < 2:
        raise ReplayCompatibilityError(
            "compact root tape comparison requires at least two services"
        )
    reference_key = str(reference_label)
    if reference_key not in service_map:
        raise ReplayCompatibilityError("reference_label must name one service")

    backend_totals: dict[str, dict[str, Any]] = {
        label: {
            "run_count": 0,
            "active_root_count": 0,
            "run_sec": 0.0,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "model_compile_requested_count": 0,
            "model_compile_used_count": 0,
            "model_compile_cache_hit_count": 0,
            "model_compile_runtime_status_counts": {},
        }
        for label in service_map
    }
    comparison_totals: dict[str, dict[str, Any]] = {}
    per_record: list[dict[str, Any]] = []

    for record in tape.records:
        results: dict[str, CompactSearchResultV1] = {}
        record_backends: dict[str, dict[str, Any]] = {}
        for label, service in service_map.items():
            root_batch = compact_root_batch_v1_from_tape_record(record)
            started = time.perf_counter()
            result = service.run(root_batch)
            run_sec = time.perf_counter() - started
            validate_compact_search_result_identity_v1(root_batch, result)
            results[label] = result
            transfer = _search_result_transfer_bytes(result)
            compile_telemetry = _model_compile_telemetry(result)
            active_count = int(result.selected_action.shape[0])
            backend_total = backend_totals[label]
            backend_total["run_count"] += 1
            backend_total["active_root_count"] += active_count
            backend_total["run_sec"] += float(run_sec)
            backend_total["h2d_bytes"] += int(transfer["h2d_bytes"])
            backend_total["d2h_bytes"] += int(transfer["d2h_bytes"])
            if compile_telemetry["requested"]:
                backend_total["model_compile_requested_count"] += 1
            if compile_telemetry["used"]:
                backend_total["model_compile_used_count"] += 1
            if compile_telemetry["cache_hit"]:
                backend_total["model_compile_cache_hit_count"] += 1
            status_counts = backend_total["model_compile_runtime_status_counts"]
            status = str(compile_telemetry["runtime_status"])
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            record_backends[label] = {
                "run_sec": float(run_sec),
                "active_root_count": active_count,
                "h2d_bytes": int(transfer["h2d_bytes"]),
                "d2h_bytes": int(transfer["d2h_bytes"]),
                "search_impl": str(result.metadata.get("search_impl", "")),
                "model_compile_requested": bool(compile_telemetry["requested"]),
                "model_compile_used": bool(compile_telemetry["used"]),
                "model_compile_cache_hit": bool(compile_telemetry["cache_hit"]),
                "model_compile_mode": str(compile_telemetry["mode"]),
                "model_compile_runtime_status": status,
            }

        reference = results[reference_key]
        record_comparisons: dict[str, dict[str, Any]] = {}
        for label, result in results.items():
            if label == reference_key:
                continue
            comparison_key = f"{label}_vs_{reference_key}"
            telemetry = compact_search_comparison_telemetry_v1(
                result,
                reference,
                comparison_label=comparison_key,
            )
            active = int(telemetry["compact_search_comparator_active_root_count"])
            _accumulate_comparison(comparison_totals, comparison_key, telemetry, active)
            record_comparisons[comparison_key] = telemetry

        per_record.append(
            {
                "record_index": int(record.record_index),
                "backend": record_backends,
                "comparison": record_comparisons,
            }
        )

    return {
        "schema_id": COMPACT_ROOT_TAPE_COMPARISON_SCHEMA_ID,
        "record_count": len(tape.records),
        "reference_label": reference_key,
        "tape_metadata": dict(tape.metadata),
        "backend": {
            label: {
                **stats,
                "run_sec_per_active_root": (
                    0.0
                    if int(stats["active_root_count"]) == 0
                    else float(stats["run_sec"]) / float(stats["active_root_count"])
                ),
            }
            for label, stats in backend_totals.items()
        },
        "comparison": {
            label: _finalize_comparison_totals(stats)
            for label, stats in comparison_totals.items()
        },
        "per_record": per_record,
    }


def write_compact_root_tape_npz_v1(path: str | Path, tape: CompactRootTapeV1) -> None:
    """Write a compact root tape to a compressed NPZ artifact."""

    target = Path(path)
    payload: dict[str, Any] = {
        "schema_id": np.asarray(tape.schema_id),
        "metadata_json": np.asarray(json.dumps(_plain_metadata(tape.metadata), sort_keys=True)),
        "record_count": np.asarray(len(tape.records), dtype=np.int64),
    }
    for index, record in enumerate(tape.records):
        prefix = f"record_{index}"
        payload[f"{prefix}_record_index"] = np.asarray(record.record_index, dtype=np.int64)
        payload[f"{prefix}_observation"] = record.observation
        payload[f"{prefix}_legal_mask"] = record.legal_mask
        payload[f"{prefix}_active_root_mask"] = record.active_root_mask
        payload[f"{prefix}_to_play"] = record.to_play
        payload[f"{prefix}_env_row"] = record.env_row
        payload[f"{prefix}_player"] = record.player
        payload[f"{prefix}_policy_env_id"] = record.policy_env_id
        payload[f"{prefix}_target_reward"] = record.target_reward
        payload[f"{prefix}_done_root"] = record.done_root
        payload[f"{prefix}_final_observation_present"] = np.asarray(
            record.final_observation is not None,
            dtype=np.bool_,
        )
        if record.final_observation is not None:
            payload[f"{prefix}_final_observation"] = record.final_observation
        payload[f"{prefix}_final_observation_row_mask"] = (
            record.final_observation_row_mask
        )
        payload[f"{prefix}_terminal_row_mask"] = record.terminal_row_mask
        payload[f"{prefix}_autoreset_row_mask"] = record.autoreset_row_mask
        payload[f"{prefix}_root_metadata_json"] = np.asarray(
            json.dumps(_plain_metadata(record.root_metadata), sort_keys=True)
        )
        payload[f"{prefix}_capture_metadata_json"] = np.asarray(
            json.dumps(_plain_metadata(record.capture_metadata), sort_keys=True)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)


def read_compact_root_tape_npz_v1(path: str | Path) -> CompactRootTapeV1:
    """Read a compact root tape from a compressed NPZ artifact."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        schema_id = str(data["schema_id"].item())
        if schema_id != COMPACT_ROOT_TAPE_SCHEMA_ID:
            raise ReplayCompatibilityError("unknown compact root tape NPZ schema")
        record_count = int(data["record_count"].item())
        records = []
        for index in range(record_count):
            prefix = f"record_{index}"
            final_present = bool(data[f"{prefix}_final_observation_present"].item())
            records.append(
                CompactRootTapeRecordV1(
                    schema_id=COMPACT_ROOT_TAPE_RECORD_SCHEMA_ID,
                    record_index=int(data[f"{prefix}_record_index"].item()),
                    observation=np.asarray(data[f"{prefix}_observation"]).copy(),
                    legal_mask=np.asarray(data[f"{prefix}_legal_mask"], dtype=np.bool_).copy(),
                    active_root_mask=np.asarray(
                        data[f"{prefix}_active_root_mask"],
                        dtype=np.bool_,
                    ).copy(),
                    to_play=np.asarray(data[f"{prefix}_to_play"], dtype=np.int64).copy(),
                    env_row=np.asarray(data[f"{prefix}_env_row"], dtype=np.int32).copy(),
                    player=np.asarray(data[f"{prefix}_player"], dtype=np.int16).copy(),
                    policy_env_id=np.asarray(
                        data[f"{prefix}_policy_env_id"],
                        dtype=np.int64,
                    ).copy(),
                    target_reward=np.asarray(
                        data[f"{prefix}_target_reward"],
                        dtype=np.float32,
                    ).copy(),
                    done_root=np.asarray(data[f"{prefix}_done_root"], dtype=np.bool_).copy(),
                    final_observation=None
                    if not final_present
                    else np.asarray(data[f"{prefix}_final_observation"]).copy(),
                    final_observation_row_mask=np.asarray(
                        data[f"{prefix}_final_observation_row_mask"],
                        dtype=np.bool_,
                    ).copy(),
                    terminal_row_mask=np.asarray(
                        data[f"{prefix}_terminal_row_mask"],
                        dtype=np.bool_,
                    ).copy(),
                    autoreset_row_mask=np.asarray(
                        data[f"{prefix}_autoreset_row_mask"],
                        dtype=np.bool_,
                    ).copy(),
                    root_metadata=json.loads(str(data[f"{prefix}_root_metadata_json"].item())),
                    capture_metadata=json.loads(
                        str(data[f"{prefix}_capture_metadata_json"].item())
                    ),
                )
            )
        return CompactRootTapeV1(
            schema_id=schema_id,
            records=tuple(records),
            metadata=json.loads(str(data["metadata_json"].item())),
        )


def _accumulate_comparison(
    totals: dict[str, dict[str, Any]],
    label: str,
    telemetry: Mapping[str, Any],
    active_count: int,
) -> None:
    stats = totals.setdefault(
        label,
        {
            "record_count": 0,
            "active_root_count": 0,
            "action_match_count": 0.0,
            "visit_l1_weighted_sum": 0.0,
            "visit_l1_max": 0.0,
            "root_value_abs_diff_weighted_sum": 0.0,
            "root_value_abs_diff_max": 0.0,
        },
    )
    stats["record_count"] += 1
    stats["active_root_count"] += int(active_count)
    stats["action_match_count"] += float(
        telemetry["compact_search_comparator_action_match_count"]
    )
    stats["visit_l1_weighted_sum"] += (
        float(telemetry["compact_search_comparator_visit_l1_mean"]) * active_count
    )
    stats["visit_l1_max"] = max(
        float(stats["visit_l1_max"]),
        float(telemetry["compact_search_comparator_visit_l1_max"]),
    )
    stats["root_value_abs_diff_weighted_sum"] += (
        float(telemetry["compact_search_comparator_root_value_abs_diff_mean"])
        * active_count
    )
    stats["root_value_abs_diff_max"] = max(
        float(stats["root_value_abs_diff_max"]),
        float(telemetry["compact_search_comparator_root_value_abs_diff_max"]),
    )


def _finalize_comparison_totals(stats: Mapping[str, Any]) -> dict[str, Any]:
    active = int(stats["active_root_count"])
    return {
        "record_count": int(stats["record_count"]),
        "active_root_count": active,
        "action_match_count": float(stats["action_match_count"]),
        "action_match_fraction": (
            1.0 if active == 0 else float(stats["action_match_count"]) / float(active)
        ),
        "visit_l1_mean": (
            0.0
            if active == 0
            else float(stats["visit_l1_weighted_sum"]) / float(active)
        ),
        "visit_l1_max": float(stats["visit_l1_max"]),
        "root_value_abs_diff_mean": (
            0.0
            if active == 0
            else float(stats["root_value_abs_diff_weighted_sum"]) / float(active)
        ),
        "root_value_abs_diff_max": float(stats["root_value_abs_diff_max"]),
    }


def _search_result_transfer_bytes(result: CompactSearchResultV1) -> dict[str, int]:
    telemetry = result.metadata.get("profile_telemetry", {})
    if not isinstance(telemetry, Mapping):
        return {"h2d_bytes": 0, "d2h_bytes": 0}
    h2d = 0
    d2h = 0
    for key, value in telemetry.items():
        if not str(key).endswith("_bytes"):
            continue
        try:
            bytes_value = int(value)
        except (TypeError, ValueError):
            continue
        key_text = str(key).lower()
        if "h2d" in key_text or "host_to_device" in key_text:
            h2d += bytes_value
        if (
            "d2h" in key_text
            or "device_to_host" in key_text
            or "readback" in key_text
        ):
            d2h += bytes_value
    return {"h2d_bytes": h2d, "d2h_bytes": d2h}


def _model_compile_telemetry(result: CompactSearchResultV1) -> dict[str, Any]:
    telemetry = result.metadata.get("profile_telemetry", {})
    profile = telemetry if isinstance(telemetry, Mapping) else {}

    def get(key: str, default: Any = None) -> Any:
        return result.metadata.get(key, profile.get(key, default))

    return {
        "requested": _truthy_profile_value(
            get("compact_torch_search_model_compile_requested", False)
        ),
        "used": _truthy_profile_value(
            get("compact_torch_search_model_compile_used", False)
        ),
        "cache_hit": _truthy_profile_value(
            get("compact_torch_search_model_compile_cache_hit", False)
        ),
        "mode": str(get("compact_torch_search_model_compile_mode", "none")),
        "runtime_status": str(
            get("compact_torch_search_model_compile_runtime_status", "unknown")
        ),
    }


def _truthy_profile_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return float(value) != 0.0
    except (TypeError, ValueError):
        return str(value).strip().lower() in {"true", "yes", "on"}


def _plain_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(value) for key, value in metadata.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _positive_int(value: int | None) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ReplayCompatibilityError("max_records must be positive when provided")
    return parsed


__all__ = [
    "COMPACT_ROOT_TAPE_COMPARISON_SCHEMA_ID",
    "COMPACT_ROOT_TAPE_RECORD_SCHEMA_ID",
    "COMPACT_ROOT_TAPE_SCHEMA_ID",
    "CompactRootTapeRecordV1",
    "CompactRootTapeV1",
    "InMemoryCompactRootTapeRecorderV1",
    "compact_root_batch_v1_from_tape_record",
    "compact_root_tape_record_v1_from_root_batch",
    "read_compact_root_tape_npz_v1",
    "run_compact_root_tape_comparison_v1",
    "write_compact_root_tape_npz_v1",
]
