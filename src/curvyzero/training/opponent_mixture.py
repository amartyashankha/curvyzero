"""Episode-level opponent mixture parsing and deterministic selection."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any


OPPONENT_MIXTURE_SCHEMA_ID = "curvyzero_episode_opponent_mixture/v0"
OPPONENT_MIXTURE_SELECTION_UNIT = "episode_reset"
OPPONENT_SPLIT_PLAN_SCHEMA_ID = "curvyzero_collector_opponent_split_plan/v0"
OPPONENT_SPLIT_UNIT_COLLECTOR_ENV = "collector_env"
OPPONENT_SPLIT_MODE_EXPLICIT_SLOT_COUNT_BAG_SHUFFLED = "explicit_slot_count_bag_shuffled"

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
ALLOWED_ENTRY_KEYS = {
    "name",
    "weight",
    "age_label",
    "tags",
    "opponent_policy_kind",
    "opponent_runtime_mode",
    "opponent_immortal",
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


def deterministic_collector_env_mixture_plan(
    mixture: dict[str, Any],
    *,
    env_num: int,
    seed_context: Any | None = None,
) -> dict[str, Any]:
    """Resolve explicit slot counts into deterministic collector env assignments."""

    env_count = int(env_num)
    if env_count < 1:
        raise ValueError("env_num must be at least 1")
    validated = validate_opponent_mixture(mixture)
    slot_counts_raw = _explicit_slot_counts_from_entry_weights(validated["entries"])
    slot_count_total = sum(slot_counts_raw)
    if not _is_power_of_two(slot_count_total):
        raise ValueError(
            "deterministic opponent split slot counts must sum to a power of two; "
            f"got {slot_count_total}"
        )
    if slot_count_total > env_count:
        raise ValueError(
            "deterministic opponent split slot count total cannot exceed env_num; "
            f"got slot_count_total={slot_count_total}, env_num={env_count}"
        )
    if env_count % slot_count_total != 0:
        raise ValueError(
            "deterministic opponent split slot count total must divide env_num; "
            f"got slot_count_total={slot_count_total}, env_num={env_count}"
        )
    repetition_count = env_count // slot_count_total
    entry_indices: list[int] = []
    for entry_index, slot_count in enumerate(slot_counts_raw):
        entry_indices.extend([entry_index] * slot_count * repetition_count)
    rng = random.Random(
        _split_plan_seed(
            mixture=validated,
            env_num=env_count,
            seed_context=seed_context,
            slot_count_total=slot_count_total,
        )
    )
    rng.shuffle(entry_indices)
    assignments = [
        {
            "env_id": env_id,
            "entry_index": entry_index,
            "entry_name": validated["entries"][entry_index]["name"],
        }
        for env_id, entry_index in enumerate(entry_indices)
    ]
    slot_counts = [
        {
            "entry_index": entry_index,
            "entry_name": entry["name"],
            "slot_count": int(slot_counts_raw[entry_index]),
            "slot_fraction": float(slot_counts_raw[entry_index]) / float(slot_count_total),
            "count": int(slot_counts_raw[entry_index] * repetition_count),
            "actual_fraction": float(slot_counts_raw[entry_index] * repetition_count)
            / float(env_count),
        }
        for entry_index, entry in enumerate(validated["entries"])
    ]
    plan_payload = {
        "schema_id": OPPONENT_SPLIT_PLAN_SCHEMA_ID,
        "unit": OPPONENT_SPLIT_UNIT_COLLECTOR_ENV,
        "mode": OPPONENT_SPLIT_MODE_EXPLICIT_SLOT_COUNT_BAG_SHUFFLED,
        "env_num": env_count,
        "slot_count_total": int(slot_count_total),
        "repetition_count": int(repetition_count),
        "mixture_seed": int(validated.get("seed", 0)),
        "seed_context": seed_context,
        "slot_counts": slot_counts,
        "assignments": assignments,
    }
    plan_payload["plan_sha256"] = _json_sha256(plan_payload)
    return plan_payload


def singleton_mixture_for_split_entry(
    mixture: dict[str, Any],
    *,
    entry_index: int,
) -> dict[str, Any]:
    """Return a mixture that always selects one entry from a validated mixture."""

    validated = validate_opponent_mixture(mixture)
    index = int(entry_index)
    if index < 0 or index >= len(validated["entries"]):
        raise IndexError(f"opponent mixture entry_index out of range: {entry_index}")
    entry = dict(validated["entries"][index])
    entry["weight"] = 1.0
    return validate_opponent_mixture(
        {
            "schema_id": OPPONENT_MIXTURE_SCHEMA_ID,
            "selection_unit": OPPONENT_MIXTURE_SELECTION_UNIT,
            "seed": int(validated.get("seed", 0)),
            "entries": [entry],
        }
    )


def _validate_entry(raw_entry: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(raw_entry, dict):
        raise ValueError(f"opponent mixture entry {index} must be an object")
    if "opponent_death_mode" in raw_entry:
        name = raw_entry.get("name", index)
        raise ValueError(
            f"opponent mixture entry {name!r} must use opponent_immortal; "
            "opponent_death_mode is derived runtime metadata"
        )
    unknown = set(raw_entry) - ALLOWED_ENTRY_KEYS
    if unknown:
        raise ValueError(f"opponent mixture entry {index} has unknown keys {sorted(unknown)!r}")
    name = raw_entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"opponent mixture entry {index} requires non-empty name")
    if "weight" not in raw_entry:
        raise ValueError(f"opponent mixture entry {name!r} requires weight")
    weight = float(raw_entry["weight"])
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"opponent mixture entry {name!r} weight must be positive")
    if "opponent_policy_kind" not in raw_entry:
        raise ValueError(f"opponent mixture entry {name!r} requires opponent_policy_kind")
    policy_kind = str(raw_entry["opponent_policy_kind"])
    if policy_kind not in SUPPORTED_OPPONENT_POLICY_KINDS:
        raise ValueError(
            f"opponent mixture entry {name!r} has unsupported opponent_policy_kind {policy_kind!r}"
        )
    runtime_mode = str(raw_entry.get("opponent_runtime_mode", OPPONENT_RUNTIME_MODE_NORMAL))
    if runtime_mode not in SUPPORTED_OPPONENT_RUNTIME_MODES:
        raise ValueError(
            f"opponent mixture entry {name!r} has unsupported opponent_runtime_mode "
            f"{runtime_mode!r}"
        )
    raw_immortal = raw_entry.get("opponent_immortal", False)
    if isinstance(raw_immortal, bool):
        immortal = bool(raw_immortal)
    else:
        raise ValueError(f"opponent mixture entry {name!r} opponent_immortal must be a boolean")
    death_mode = OPPONENT_DEATH_MODE_IMMORTAL if immortal else OPPONENT_DEATH_MODE_NORMAL
    if runtime_mode == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP:
        if policy_kind != OPPONENT_POLICY_KIND_FIXED_STRAIGHT:
            raise ValueError(f"blank_canvas_noop mixture entry {name!r} must use fixed_straight")
        if not immortal:
            raise ValueError(
                f"blank_canvas_noop mixture entry {name!r} must set opponent_immortal=true"
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
            raise ValueError(f"frozen mixture entry {name!r} must use normal opponent_runtime_mode")
    elif has_checkpoint_ref or has_checkpoint_path:
        raise ValueError(f"non-frozen mixture entry {name!r} cannot set opponent checkpoint fields")
    entry = dict(raw_entry)
    entry["name"] = name.strip()
    entry["weight"] = weight
    entry["opponent_policy_kind"] = policy_kind
    entry["opponent_runtime_mode"] = runtime_mode
    entry["opponent_immortal"] = death_mode == OPPONENT_DEATH_MODE_IMMORTAL
    return entry


def _selected_entry_payload(
    entry: dict[str, Any],
    *,
    index: int,
    mixture: dict[str, Any],
) -> dict[str, Any]:
    selected = dict(entry)
    selected["opponent_death_mode"] = (
        OPPONENT_DEATH_MODE_IMMORTAL
        if bool(selected.get("opponent_immortal", False))
        else OPPONENT_DEATH_MODE_NORMAL
    )
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


def _explicit_slot_counts_from_entry_weights(entries: list[dict[str, Any]]) -> list[int]:
    slot_counts: list[int] = []
    for entry in entries:
        weight = float(entry["weight"])
        slot_count = int(weight)
        if not math.isfinite(weight) or weight != float(slot_count):
            raise ValueError(
                "deterministic opponent split requires integer entry weights as "
                f"slot counts; entry {entry['name']!r} has weight {entry['weight']!r}"
            )
        if slot_count < 1:
            raise ValueError(
                "deterministic opponent split entry slot counts must be positive; "
                f"entry {entry['name']!r} has {slot_count}"
            )
        slot_counts.append(slot_count)
    return slot_counts


def _is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def _split_plan_seed(
    *,
    mixture: dict[str, Any],
    env_num: int,
    seed_context: Any | None,
    slot_count_total: int,
) -> int:
    payload = {
        "schema_id": OPPONENT_SPLIT_PLAN_SCHEMA_ID,
        "mode": OPPONENT_SPLIT_MODE_EXPLICIT_SLOT_COUNT_BAG_SHUFFLED,
        "mixture_schema_id": OPPONENT_MIXTURE_SCHEMA_ID,
        "mixture_seed": int(mixture.get("seed", 0)),
        "env_num": int(env_num),
        "slot_count_total": int(slot_count_total),
        "seed_context": seed_context,
        "entries": [
            {"name": entry["name"], "weight": float(entry["weight"])}
            for entry in mixture["entries"]
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
