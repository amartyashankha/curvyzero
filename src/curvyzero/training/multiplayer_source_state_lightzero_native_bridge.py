"""Opt-in LightZero ``GameSegment`` construction smoke for source-state rows.

This module sits beside the injection-only source-state bridge. It imports
LightZero lazily inside the build/push helpers and keeps metadata claims false
until a separate sampled-target parity proof exists.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from curvyzero.training.multiplayer_source_state_native_bridge import (
    SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_CONTRACT_ID,
)
from curvyzero.training.multiplayer_source_state_native_bridge import (
    SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_SCHEMA_ID,
)
from curvyzero.training.multiplayer_source_state_native_bridge import (
    SOURCE_STATE_NATIVE_GAME_SEGMENTS_NON_CLAIMS,
)
from curvyzero.training.multiplayer_source_state_native_bridge import (
    SourceStateNativeGameSegmentSpecV0,
)
from curvyzero.training.multiplayer_source_state_native_bridge import (
    build_source_state_native_game_segment_specs_v0,
)
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
from curvyzero.training.two_seat_native_replay_bridge import (
    NativeBufferPushResult,
    NativeReplayBridgeUnavailable,
    build_lightzero_muzero_bridge_config,
    push_native_segments_into_muzero_game_buffer,
)


LIGHTZERO_SOURCE_STATE_NATIVE_GAME_SEGMENTS_CONTRACT_ID = (
    "curvyzero_lightzero_source_state_native_game_segments/v0"
)
LIGHTZERO_SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA_ID = (
    "curvyzero_lightzero_source_state_native_game_segments_schema/v0"
)
LIGHTZERO_SOURCE_STATE_NATIVE_GAME_SEGMENTS_METADATA_SCHEMA_ID = (
    "curvyzero_lightzero_source_state_native_game_segments_metadata/v0"
)


@dataclass(frozen=True, slots=True)
class LightZeroSourceStateNativeGameSegmentsV0:
    """Source-state specs and opt-in LightZero-style ``GameSegment`` objects."""

    metadata: dict[str, Any]
    specs: tuple[SourceStateNativeGameSegmentSpecV0, ...]
    game_segments: tuple[Any, ...]
    config: Any


def build_lightzero_source_state_native_game_segments_v0(
    target_rows_or_specs: (
        SourceStateMultiplayerTargetRowsV0
        | SourceStateNativeGameSegmentSpecV0
        | Sequence[SourceStateNativeGameSegmentSpecV0]
    ),
    *,
    game_segment_cls: Callable[..., Any] | None = None,
    config: Any | None = None,
    action_space_size: int = 3,
) -> LightZeroSourceStateNativeGameSegmentsV0:
    """Build opt-in LightZero ``GameSegment`` objects as construction smoke only."""

    if game_segment_cls is None:
        try:
            from lzero.mcts.buffer.game_segment import GameSegment
        except Exception as exc:
            raise NativeReplayBridgeUnavailable(
                f"LightZero GameSegment import failed: {exc}"
            ) from exc
        game_segment_cls = GameSegment

    specs, source_metadata = _resolve_specs_and_metadata(target_rows_or_specs)
    max_segment_length = max((len(spec.actions) for spec in specs), default=0)
    resolved_config = config
    if resolved_config is None:
        resolved_config = build_lightzero_muzero_bridge_config(
            action_space_size=action_space_size,
            game_segment_length=max_segment_length,
            num_unroll_steps=max(0, max_segment_length - 1),
            td_steps=max_segment_length,
        )

    action_space = _DiscreteActionSpace(int(action_space_size))
    game_segments = tuple(
        _build_lightzero_game_segment(
            spec,
            game_segment_cls=game_segment_cls,
            action_space=action_space,
            config=resolved_config,
        )
        for spec in specs
    )
    return LightZeroSourceStateNativeGameSegmentsV0(
        metadata=_metadata(
            source_metadata=source_metadata,
            segment_count=len(specs),
            action_space_size=action_space_size,
        ),
        specs=specs,
        game_segments=game_segments,
        config=resolved_config,
    )


def maybe_push_lightzero_source_state_native_segments_into_muzero_buffer_v0(
    result: LightZeroSourceStateNativeGameSegmentsV0,
    *,
    buffer_cls: type[Any] | None = None,
) -> NativeBufferPushResult:
    """Push constructed source-state segments into ``MuZeroGameBuffer`` if available."""

    return push_native_segments_into_muzero_game_buffer(
        result.game_segments,
        config=result.config,
        buffer_cls=buffer_cls,
    )


def _resolve_specs_and_metadata(
    target_rows_or_specs: (
        SourceStateMultiplayerTargetRowsV0
        | SourceStateNativeGameSegmentSpecV0
        | Sequence[SourceStateNativeGameSegmentSpecV0]
    ),
) -> tuple[tuple[SourceStateNativeGameSegmentSpecV0, ...], Mapping[str, Any]]:
    if isinstance(target_rows_or_specs, SourceStateMultiplayerTargetRowsV0):
        return (
            build_source_state_native_game_segment_specs_v0(target_rows_or_specs),
            target_rows_or_specs.metadata,
        )
    if isinstance(target_rows_or_specs, SourceStateNativeGameSegmentSpecV0):
        return ((target_rows_or_specs,), {})
    return (tuple(target_rows_or_specs), {})


def _build_lightzero_game_segment(
    spec: SourceStateNativeGameSegmentSpecV0,
    *,
    game_segment_cls: Callable[..., Any],
    action_space: Any,
    config: Any,
) -> Any:
    segment = game_segment_cls(
        action_space,
        game_segment_length=len(spec.actions),
        config=config,
    )
    segment.reset([np.asarray(spec.observations[0], dtype=np.float32).copy()])
    for index, action in enumerate(spec.actions):
        segment.store_search_stats(
            list(np.asarray(spec.visit_distributions[index], dtype=np.float32)),
            root_value=float(spec.root_values[index]),
        )
        segment.append(
            int(action),
            np.asarray(spec.observations[index + 1], dtype=np.float32).copy(),
            float(spec.rewards[index]),
            action_mask=np.asarray(spec.action_masks[index], dtype=np.int8).copy(),
            to_play=int(spec.to_play[index]),
            timestep=int(spec.record_indices[index]),
        )
    if hasattr(segment, "game_segment_to_array"):
        segment.game_segment_to_array()
    return segment


def _metadata(
    *,
    source_metadata: Mapping[str, Any],
    segment_count: int,
    action_space_size: int,
) -> dict[str, Any]:
    metadata = dict(source_metadata)
    metadata.update(
        {
            "metadata_schema_id": LIGHTZERO_SOURCE_STATE_NATIVE_GAME_SEGMENTS_METADATA_SCHEMA_ID,
            "bridge_contract_id": LIGHTZERO_SOURCE_STATE_NATIVE_GAME_SEGMENTS_CONTRACT_ID,
            "bridge_schema_id": LIGHTZERO_SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA_ID,
            "spec_contract_id": SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_CONTRACT_ID,
            "spec_schema_id": SOURCE_STATE_NATIVE_GAME_SEGMENT_SPECS_SCHEMA_ID,
            "source_target_contract_id": source_metadata.get(
                "target_contract_id",
                SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
            ),
            "source_target_schema_id": source_metadata.get(
                "target_schema_id",
                SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
            ),
            "source_target_schema_hash": source_metadata.get(
                "target_schema_hash",
                SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH,
            ),
            "segment_count": int(segment_count),
            "action_space_size": int(action_space_size),
            "integration_policy": "opt_in_lightzero_construction_smoke_only/v0",
            "construction_smoke": True,
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
        "project_training_helper_metadata" in source_metadata
        and isinstance(source_metadata["project_training_helper_metadata"], Mapping)
    ):
        metadata["project_training_helper_metadata"] = dict(
            source_metadata["project_training_helper_metadata"]
        )
    return metadata


@dataclass(frozen=True)
class _DiscreteActionSpace:
    n: int


__all__ = [
    "LIGHTZERO_SOURCE_STATE_NATIVE_GAME_SEGMENTS_CONTRACT_ID",
    "LIGHTZERO_SOURCE_STATE_NATIVE_GAME_SEGMENTS_METADATA_SCHEMA_ID",
    "LIGHTZERO_SOURCE_STATE_NATIVE_GAME_SEGMENTS_SCHEMA_ID",
    "LightZeroSourceStateNativeGameSegmentsV0",
    "build_lightzero_source_state_native_game_segments_v0",
    "maybe_push_lightzero_source_state_native_segments_into_muzero_buffer_v0",
]
