"""Pure opponent assignment helpers for training launch inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from curvyzero.training.lightzero_checkpoints import (
    lightzero_iteration_from_checkpoint_name,
)
from curvyzero.training.opponent_mixture import (
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    validate_opponent_mixture,
)

OPPONENT_ASSIGNMENT_SCHEMA_ID = "curvyzero_opponent_assignment/v0"
MUTABLE_CHECKPOINT_NAMES = {"latest.pth.tar", "ckpt_best.pth.tar"}


def parse_opponent_assignment_snapshot(value: Any) -> dict[str, Any] | None:
    """Parse a frozen opponent assignment snapshot.

    This is intentionally plain data. It does not read Modal Dicts, reload
    volumes, rank policies, or load checkpoints.
    """

    if value is None:
        return None
    parsed = _parse_assignment_json_object(value)

    unknown = set(parsed) - {
        "schema_id",
        "assignment_id",
        "source_epoch",
        "source_ref",
        "created_at",
        "seed",
        "entries",
    }
    if unknown:
        raise ValueError(
            "opponent assignment snapshot has unknown keys "
            f"{sorted(unknown)!r}"
        )

    schema_id = parsed.get("schema_id")
    if schema_id != OPPONENT_ASSIGNMENT_SCHEMA_ID:
        raise ValueError(
            "opponent assignment snapshot requires schema_id "
            f"{OPPONENT_ASSIGNMENT_SCHEMA_ID!r}"
        )

    assignment_id = parsed.get("assignment_id")
    if not isinstance(assignment_id, str) or not assignment_id.strip():
        raise ValueError("opponent assignment snapshot requires assignment_id")

    mixture = validate_opponent_mixture(
        {
            "seed": int(parsed.get("seed", 0)),
            "entries": parsed.get("entries"),
        }
    )
    _validate_assignment_frozen_refs(mixture)

    return {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": assignment_id.strip(),
        "source_epoch": parsed.get("source_epoch"),
        "source_ref": parsed.get("source_ref"),
        "created_at": parsed.get("created_at"),
        "opponent_mixture": mixture,
    }


def canonical_assignment_json_sha256(value: Any) -> str:
    """Return a stable hash for one opponent assignment JSON object."""

    parsed = _parse_assignment_json_object(value)
    canonical = json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_assignment_json_object(value: Any) -> dict[str, Any]:
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("opponent assignment snapshot must be a JSON object")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"opponent assignment snapshot must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("opponent assignment snapshot must be a JSON object")
    return parsed


def _validate_assignment_frozen_refs(mixture: dict[str, Any]) -> None:
    for entry in mixture["entries"]:
        if entry["opponent_policy_kind"] != OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
            continue
        checkpoint_ref = entry.get("opponent_checkpoint_ref")
        if not checkpoint_ref:
            raise ValueError(
                "opponent assignment frozen entries require immutable "
                "opponent_checkpoint_ref"
            )
        checkpoint_name = Path(str(checkpoint_ref)).name
        if (
            checkpoint_name in MUTABLE_CHECKPOINT_NAMES
            or lightzero_iteration_from_checkpoint_name(checkpoint_name) is None
        ):
            raise ValueError(
                "opponent assignment frozen refs must be immutable exact "
                f"iteration_N.pth.tar files; got {checkpoint_ref!r}"
            )
