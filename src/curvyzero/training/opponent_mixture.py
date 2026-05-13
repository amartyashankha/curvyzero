"""Episode-level opponent mixture parsing and deterministic selection."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any


OPPONENT_MIXTURE_SCHEMA_ID = "curvyzero_episode_opponent_mixture/v0"
OPPONENT_MIXTURE_SELECTION_UNIT = "episode_reset"

OPPONENT_POLICY_KIND_FIXED_STRAIGHT = "fixed_straight"
OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT = "proactive_wall_avoidant"
OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT = "frozen_lightzero_checkpoint"
OPPONENT_RUNTIME_MODE_NORMAL = "normal"
OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP = "blank_canvas_noop"
OPPONENT_DEATH_MODE_NORMAL = "normal"
OPPONENT_DEATH_MODE_IMMORTAL = "immortal"

SUPPORTED_OPPONENT_POLICY_KINDS = (
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
)
SUPPORTED_OPPONENT_RUNTIME_MODES = (
    OPPONENT_RUNTIME_MODE_NORMAL,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
)
SUPPORTED_OPPONENT_DEATH_MODES = (
    OPPONENT_DEATH_MODE_NORMAL,
    OPPONENT_DEATH_MODE_IMMORTAL,
)
ALLOWED_ENTRY_KEYS = {
    "name",
    "weight",
    "age_label",
    "tags",
    "opponent_policy_kind",
    "opponent_runtime_mode",
    "opponent_death_mode",
    "opponent_checkpoint_ref",
    "opponent_checkpoint_path",
    "opponent_checkpoint_resolution",
    "opponent_checkpoint_file",
    "opponent_checkpoint_report_ref",
    "opponent_snapshot_ref",
    "opponent_checkpoint_state_key",
    "opponent_policy_seed",
    "opponent_num_simulations",
    "opponent_batch_size",
    "opponent_use_cuda",
    "opponent_wall_avoidant_safe_margin",
}


def parse_opponent_mixture_spec(value: Any) -> dict[str, Any] | None:
    """Parse and validate an explicit weighted opponent mixture spec."""

    if value is None:
        return None
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"opponent_mixture_spec must be valid JSON: {exc}") from exc
    return validate_opponent_mixture(parsed)


def validate_opponent_mixture(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        raw_entries = value
        raw_seed = 0
    elif isinstance(value, dict):
        unknown = set(value) - {
            "schema_id",
            "selection_unit",
            "seed",
            "total_weight",
            "entries",
        }
        if unknown:
            raise ValueError(
                "opponent mixture top-level keys must be schema_id, seed, entries; "
                f"got unknown keys {sorted(unknown)!r}"
            )
        raw_entries = value.get("entries")
        raw_seed = value.get("seed", 0)
    else:
        raise ValueError("opponent mixture must be a JSON object or list")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("opponent mixture entries must be a non-empty list")

    entries: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    total_weight = 0.0
    for index, raw_entry in enumerate(raw_entries):
        entry = _validate_entry(raw_entry, index=index)
        if entry["name"] in seen_names:
            raise ValueError(f"opponent mixture entry name is duplicated: {entry['name']!r}")
        seen_names.add(entry["name"])
        total_weight += float(entry["weight"])
        entries.append(entry)
    if total_weight <= 0.0:
        raise ValueError("opponent mixture total weight must be positive")
    return {
        "schema_id": OPPONENT_MIXTURE_SCHEMA_ID,
        "selection_unit": OPPONENT_MIXTURE_SELECTION_UNIT,
        "seed": int(raw_seed),
        "total_weight": float(total_weight),
        "entries": entries,
    }


def select_opponent_mixture_entry(
    mixture: dict[str, Any],
    *,
    episode_seed: int,
    reset_index: int,
) -> dict[str, Any]:
    """Select one entry deterministically for a reset/episode boundary."""

    validated = validate_opponent_mixture(mixture)
    rng = random.Random(
        _selection_seed(
            mixture_seed=int(validated.get("seed", 0)),
            episode_seed=int(episode_seed),
            reset_index=int(reset_index),
        )
    )
    threshold = rng.random() * float(validated["total_weight"])
    cumulative = 0.0
    for index, entry in enumerate(validated["entries"]):
        cumulative += float(entry["weight"])
        if threshold < cumulative:
            return _selected_entry_payload(entry, index=index, mixture=validated)
    raise RuntimeError("opponent mixture selection failed despite positive total weight")


def _validate_entry(raw_entry: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(raw_entry, dict):
        raise ValueError(f"opponent mixture entry {index} must be an object")
    unknown = set(raw_entry) - ALLOWED_ENTRY_KEYS
    if unknown:
        raise ValueError(
            f"opponent mixture entry {index} has unknown keys {sorted(unknown)!r}"
        )
    name = raw_entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"opponent mixture entry {index} requires non-empty name")
    if "weight" not in raw_entry:
        raise ValueError(f"opponent mixture entry {name!r} requires weight")
    weight = float(raw_entry["weight"])
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"opponent mixture entry {name!r} weight must be positive")
    if "opponent_policy_kind" not in raw_entry:
        raise ValueError(
            f"opponent mixture entry {name!r} requires opponent_policy_kind"
        )
    policy_kind = str(raw_entry["opponent_policy_kind"])
    if policy_kind not in SUPPORTED_OPPONENT_POLICY_KINDS:
        raise ValueError(
            f"opponent mixture entry {name!r} has unsupported opponent_policy_kind "
            f"{policy_kind!r}"
        )
    runtime_mode = str(raw_entry.get("opponent_runtime_mode", OPPONENT_RUNTIME_MODE_NORMAL))
    if runtime_mode not in SUPPORTED_OPPONENT_RUNTIME_MODES:
        raise ValueError(
            f"opponent mixture entry {name!r} has unsupported opponent_runtime_mode "
            f"{runtime_mode!r}"
        )
    death_mode = str(raw_entry.get("opponent_death_mode", OPPONENT_DEATH_MODE_NORMAL))
    if death_mode not in SUPPORTED_OPPONENT_DEATH_MODES:
        raise ValueError(
            f"opponent mixture entry {name!r} has unsupported opponent_death_mode "
            f"{death_mode!r}"
        )
    if runtime_mode == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP:
        if policy_kind != OPPONENT_POLICY_KIND_FIXED_STRAIGHT:
            raise ValueError(
                f"blank_canvas_noop mixture entry {name!r} must use fixed_straight"
            )
    has_checkpoint_ref = bool(raw_entry.get("opponent_checkpoint_ref"))
    has_checkpoint_path = bool(raw_entry.get("opponent_checkpoint_path"))
    if policy_kind == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
        if not (has_checkpoint_ref or has_checkpoint_path):
            raise ValueError(
                f"frozen mixture entry {name!r} requires opponent_checkpoint_ref "
                "or resolved opponent_checkpoint_path"
            )
        if runtime_mode != OPPONENT_RUNTIME_MODE_NORMAL:
            raise ValueError(
                f"frozen mixture entry {name!r} must use normal opponent_runtime_mode"
            )
    elif has_checkpoint_ref or has_checkpoint_path:
        raise ValueError(
            f"non-frozen mixture entry {name!r} cannot set opponent checkpoint fields"
        )
    entry = dict(raw_entry)
    entry["name"] = name.strip()
    entry["weight"] = weight
    entry["opponent_policy_kind"] = policy_kind
    entry["opponent_runtime_mode"] = runtime_mode
    entry["opponent_death_mode"] = death_mode
    return entry


def _selected_entry_payload(
    entry: dict[str, Any],
    *,
    index: int,
    mixture: dict[str, Any],
) -> dict[str, Any]:
    selected = dict(entry)
    selected["selection_index"] = int(index)
    selected["selection_unit"] = OPPONENT_MIXTURE_SELECTION_UNIT
    selected["mixture_schema_id"] = OPPONENT_MIXTURE_SCHEMA_ID
    selected["mixture_seed"] = int(mixture.get("seed", 0))
    selected["mixture_total_weight"] = float(mixture["total_weight"])
    return selected


def _selection_seed(*, mixture_seed: int, episode_seed: int, reset_index: int) -> int:
    payload = json.dumps(
        {
            "schema_id": OPPONENT_MIXTURE_SCHEMA_ID,
            "mixture_seed": int(mixture_seed),
            "episode_seed": int(episode_seed),
            "reset_index": int(reset_index),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
