"""Injection-only native bridge spike for source-state multiplayer target rows.

This module converts repo-owned source-state target rows into per-seat
``GameSegment``-like calls supplied by the caller. It deliberately does not
import LightZero or claim that the resulting objects prove native LightZero
training compatibility.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from curvyzero.training.multiplayer_source_state_target_rows import DEFAULT_TO_PLAY
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SourceStateMultiplayerTargetRowsV0,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError
from curvyzero.training.replay_chunk_v0 import stable_contract_hash


SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_CONTRACT_ID = (
    "curvyzero_source_state_native_game_segment_specs/v0"
)
SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_SCHEMA_ID = (
    "curvyzero_source_state_native_game_segment_specs_schema/v0"
)
SOURCE_STATE_NATIVE_GAME_SEGMENTS_CONTRACT_ID = (
    "curvyzero_source_state_native_game_segments/v0"
)
SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA_ID = (
    "curvyzero_source_state_native_game_segments_schema/v0"
)
SOURCE_STATE_NATIVE_GAME_SEGMENTS_METADATA_SCHEMA_ID = (
    "curvyzero_source_state_native_game_segments_metadata/v0"
)
SOURCE_STATE_NATIVE_GAME_SEGMENTS_NON_CLAIMS = (
    "not_lightzero_training_integration",
    "not_lightzero_native_game_segment",
    "not_native_game_segment_proof",
    "not_muzero_game_buffer",
    "not_learner_update",
    "not_policy_improvement_claim",
)
SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA = {
    "schema_id": SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA_ID,
    "metadata_schema_id": SOURCE_STATE_NATIVE_GAME_SEGMENTS_METADATA_SCHEMA_ID,
    "contract_id": SOURCE_STATE_NATIVE_GAME_SEGMENTS_CONTRACT_ID,
    "spec_contract_id": SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_CONTRACT_ID,
    "spec_schema_id": SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_SCHEMA_ID,
    "source_target_contract_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
    "source_target_schema_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
    "source_target_schema_hash": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH,
    "to_play_policy": "always_-1_non_board_game_seat_perspective/v0",
    "integration_policy": "caller_injected_game_segment_class_only/v0",
    "fields": {
        "env_row": ("int32", ("segment",)),
        "player": ("int16", ("segment",)),
        "row_id": ("int64", ("segment", "transition")),
        "observations": ("float32", ("segment", "transition_plus_one", "...")),
        "actions": ("int16", ("segment", "transition")),
        "rewards": ("float32", ("segment", "transition")),
        "visit_distributions": ("float32", ("segment", "transition", "action")),
        "policy_target": ("float32", ("segment", "transition", "action")),
        "root_values": ("float32", ("segment", "transition")),
        "action_masks": ("bool", ("segment", "transition", "action")),
        "to_play": ("int64", ("segment", "transition")),
        "record_indices": ("int32", ("segment", "transition")),
        "terminal": ("bool", ("segment",)),
    },
    "non_claims": SOURCE_STATE_NATIVE_GAME_SEGMENTS_NON_CLAIMS,
}
SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA_HASH = stable_contract_hash(
    SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA
)


@dataclass(frozen=True, slots=True)
class SourceStateNativeGameSegmentSpecV0:
    """Per-seat transition sequence ready for a caller-injected segment class."""

    env_row: int
    player: int
    row_id: tuple[int, ...]
    observations: tuple[np.ndarray, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    visit_distributions: tuple[np.ndarray, ...]
    policy_target: tuple[np.ndarray, ...]
    root_values: tuple[float, ...]
    action_masks: tuple[np.ndarray, ...]
    to_play: tuple[int, ...]
    record_indices: tuple[int, ...]
    terminal: bool


@dataclass(frozen=True, slots=True)
class SourceStateNativeGameSegmentsV0:
    """Specs and injected game segment objects built from source-state rows."""

    metadata: dict[str, Any]
    specs: tuple[SourceStateNativeGameSegmentSpecV0, ...]
    game_segments: tuple[Any, ...]
    config: Any


def build_source_state_native_game_segment_specs_v0(
    target_rows: SourceStateMultiplayerTargetRowsV0,
) -> tuple[SourceStateNativeGameSegmentSpecV0, ...]:
    """Group target rows into per-env/per-player native-like segment specs."""

    row_count = _validate_target_rows(target_rows)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row_id in range(row_count):
        groups[(int(target_rows.env_row[row_id]), int(target_rows.player[row_id]))].append(
            row_id
        )

    specs: list[SourceStateNativeGameSegmentSpecV0] = []
    for env_row, player in sorted(groups):
        row_ids = tuple(
            sorted(groups[(env_row, player)], key=lambda index: int(target_rows.record_index[index]))
        )
        observations = [np.asarray(target_rows.observation[row_ids[0]], dtype=np.float32).copy()]
        observations.extend(
            np.asarray(target_rows.next_observation[row_id], dtype=np.float32).copy()
            for row_id in row_ids
        )
        policy_target = tuple(
            np.asarray(target_rows.policy_target[row_id], dtype=np.float32).copy()
            for row_id in row_ids
        )
        specs.append(
            SourceStateNativeGameSegmentSpecV0(
                env_row=env_row,
                player=player,
                row_id=tuple(int(row_id) for row_id in row_ids),
                observations=tuple(observations),
                actions=tuple(int(target_rows.action[row_id]) for row_id in row_ids),
                rewards=tuple(float(target_rows.reward[row_id]) for row_id in row_ids),
                visit_distributions=policy_target,
                policy_target=policy_target,
                root_values=tuple(
                    float(target_rows.root_value[row_id]) for row_id in row_ids
                ),
                action_masks=tuple(
                    np.asarray(target_rows.action_mask[row_id], dtype=bool).copy()
                    for row_id in row_ids
                ),
                to_play=tuple(int(target_rows.to_play[row_id]) for row_id in row_ids),
                record_indices=tuple(
                    int(target_rows.record_index[row_id]) for row_id in row_ids
                ),
                terminal=bool(np.asarray(target_rows.done, dtype=bool)[list(row_ids)].any()),
            )
        )
    return tuple(specs)


def build_source_state_native_game_segments_v0(
    target_rows: SourceStateMultiplayerTargetRowsV0,
    *,
    game_segment_cls: Callable[..., Any],
    config: Any | None = None,
) -> SourceStateNativeGameSegmentsV0:
    """Build injected fake/native segment objects from source-state target rows."""

    specs = build_source_state_native_game_segment_specs_v0(target_rows)
    game_segments = tuple(
        _build_injected_game_segment(
            spec,
            game_segment_cls=game_segment_cls,
            config=config,
        )
        for spec in specs
    )
    return SourceStateNativeGameSegmentsV0(
        metadata=_metadata(target_rows=target_rows, segment_count=len(specs)),
        specs=specs,
        game_segments=game_segments,
        config=config,
    )


def _build_injected_game_segment(
    spec: SourceStateNativeGameSegmentSpecV0,
    *,
    game_segment_cls: Callable[..., Any],
    config: Any | None,
) -> Any:
    segment = game_segment_cls(config=config)
    segment.reset(np.asarray(spec.observations[0], dtype=np.float32).copy())
    for index, action in enumerate(spec.actions):
        segment.store_search_stats(
            np.asarray(spec.visit_distributions[index], dtype=np.float32).copy(),
            root_value=float(spec.root_values[index]),
        )
        segment.append(
            int(action),
            np.asarray(spec.observations[index + 1], dtype=np.float32).copy(),
            float(spec.rewards[index]),
            action_mask=np.asarray(spec.action_masks[index], dtype=bool).copy(),
            to_play=int(spec.to_play[index]),
            timestep=int(spec.record_indices[index]),
        )
    return segment


def _validate_target_rows(target_rows: SourceStateMultiplayerTargetRowsV0) -> int:
    arrays = {
        "observation": np.asarray(target_rows.observation),
        "action": np.asarray(target_rows.action),
        "action_mask": np.asarray(target_rows.action_mask),
        "policy_target": np.asarray(target_rows.policy_target),
        "root_value": np.asarray(target_rows.root_value),
        "reward": np.asarray(target_rows.reward),
        "done": np.asarray(target_rows.done),
        "next_observation": np.asarray(target_rows.next_observation),
        "to_play": np.asarray(target_rows.to_play),
        "env_row": np.asarray(target_rows.env_row),
        "player": np.asarray(target_rows.player),
        "record_index": np.asarray(target_rows.record_index),
    }
    row_count = int(arrays["action"].shape[0])
    for name, array in arrays.items():
        if array.shape[:1] != (row_count,):
            raise ReplayCompatibilityError(f"{name} must be target-row indexed")
    if row_count and not np.array_equal(
        arrays["to_play"],
        np.full(row_count, DEFAULT_TO_PLAY, dtype=arrays["to_play"].dtype),
    ):
        raise ReplayCompatibilityError("source-state native bridge requires to_play == -1")
    return row_count


def _metadata(
    *,
    target_rows: SourceStateMultiplayerTargetRowsV0,
    segment_count: int,
) -> dict[str, Any]:
    metadata = dict(target_rows.metadata)
    metadata.update(
        {
            "metadata_schema_id": SOURCE_STATE_NATIVE_GAME_SEGMENTS_METADATA_SCHEMA_ID,
            "bridge_contract_id": SOURCE_STATE_NATIVE_GAME_SEGMENTS_CONTRACT_ID,
            "bridge_schema_id": SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA_ID,
            "bridge_schema_hash": SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA_HASH,
            "spec_contract_id": SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_CONTRACT_ID,
            "spec_schema_id": SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_SCHEMA_ID,
            "source_target_contract_id": target_rows.metadata.get(
                "target_contract_id",
                SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
            ),
            "source_target_schema_id": target_rows.metadata.get(
                "target_schema_id",
                SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
            ),
            "source_target_schema_hash": target_rows.metadata.get(
                "target_schema_hash",
                SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH,
            ),
            "segment_count": int(segment_count),
            "native_game_segment_claim": False,
            "lightzero_native_game_segment_claim": False,
            "lightzero_training_integration_claim": False,
            "lightzero_training_claim": False,
            "muzero_game_buffer_claim": False,
            "learner_update_claim": False,
            "policy_improvement_claim": False,
            "non_claims": SOURCE_STATE_NATIVE_GAME_SEGMENTS_NON_CLAIMS,
        }
    )
    if (
        "project_training_helper_metadata" in target_rows.metadata
        and isinstance(target_rows.metadata["project_training_helper_metadata"], Mapping)
    ):
        metadata["project_training_helper_metadata"] = dict(
            target_rows.metadata["project_training_helper_metadata"]
        )
    return metadata


__all__ = [
    "SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_CONTRACT_ID",
    "SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_SCHEMA_ID",
    "SOURCE_STATE_NATIVE_GAME_SEGMENTS_CONTRACT_ID",
    "SOURCE_STATE_NATIVE_GAME_SEGMENTS_METADATA_SCHEMA_ID",
    "SOURCE_STATE_NATIVE_GAME_SEGMENTS_NON_CLAIMS",
    "SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA_HASH",
    "SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA_ID",
    "SourceStateNativeGameSegmentSpecV0",
    "SourceStateNativeGameSegmentsV0",
    "build_source_state_native_game_segment_specs_v0",
    "build_source_state_native_game_segments_v0",
]
