#!/usr/bin/env python3
"""Summarize local CurvyTron tournament intake and rating JSON artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from curvyzero.contracts.curvytron import (
    curvytron_checkpoint_intake_dict_name,
    curvytron_checkpoint_intake_queue_name,
    curvytron_tournament_volume_name,
)

DEFAULT_TOURNAMENT_VOLUME = curvytron_tournament_volume_name()
CHECKPOINT_INTAKE_DICT_NAME = curvytron_checkpoint_intake_dict_name()
CHECKPOINT_INTAKE_QUEUE_NAME = curvytron_checkpoint_intake_queue_name()
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _load_json_with_modal_noise(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"(?m)^[ \t]*(?:\[(?=[\s{\]])|\{(?=\s*\"))", text):
        decoder = json.JSONDecoder()
        value, _end = decoder.raw_decode(text[match.start() :])
        return value
    raise ValueError(f"{path} does not contain JSON")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _count_from(value: Any, *keys: str) -> int | None:
    source = _mapping(value)
    for key in keys:
        if key in source and source[key] is not None:
            try:
                return int(source[key])
            except (TypeError, ValueError):
                return None
    return None


def _list_count(value: Any, *keys: str) -> int | None:
    source = _mapping(value)
    for key in keys:
        rows = _sequence(source.get(key))
        if rows:
            return len(rows)
    return None


def _first_int(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _clean_id(raw: Any, *, label: str) -> str:
    value = str(raw or "")
    if not SAFE_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            f"{label} must be 1-96 chars of letters, numbers, dash, underscore, or dot"
        )
    return value


def _slug(text: str, *, max_len: int = 36) -> str:
    chars = []
    for char in str(text):
        if char.isalnum() or char in {"-", "_", "."}:
            chars.append(char)
        elif char in {"/", " ", ":"}:
            chars.append("-")
    slug = "".join(chars).strip("-._")
    if not slug:
        slug = "item"
    return slug[:max_len].strip("-._") or "item"


def _short_hash(text: str, *, length: int = 10) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:length]


def intake_manifest_key(tournament_id: str, rating_run_id: str) -> str:
    return (
        "manifest:"
        f"{_clean_id(tournament_id, label='tournament_id')}:"
        f"{_clean_id(rating_run_id, label='rating_run_id')}"
    )


def intake_queue_partition(tournament_id: str, rating_run_id: str) -> str:
    clean_tournament_id = _clean_id(tournament_id, label="tournament_id")
    clean_rating_run_id = _clean_id(rating_run_id, label="rating_run_id")
    digest = _short_hash(f"{clean_tournament_id}:{clean_rating_run_id}", length=10)
    return (
        f"q:{_slug(clean_tournament_id, max_len=24)}:"
        f"{_slug(clean_rating_run_id, max_len=16)}:{digest}"
    )


def default_rating_claim_key(tournament_id: str, rating_run_id: str) -> str:
    return f"rating_claim:{intake_manifest_key(tournament_id, rating_run_id)}"


def unwrap_intake_manifest(payload: Any) -> dict[str, Any]:
    data = _mapping(payload)
    wrapped = _mapping(data.get("manifest"))
    if wrapped:
        return wrapped
    return data


def summarize_intake_manifest(payload: Any | None) -> dict[str, Any]:
    if payload is None:
        return {"present": False}
    manifest = unwrap_intake_manifest(payload)
    discovery = _mapping(manifest.get("discovery"))
    return {
        "present": True,
        "tournament_id": manifest.get("tournament_id"),
        "rating_run_id": manifest.get("rating_run_id"),
        "active": manifest.get("active"),
        "manifest_key": manifest.get("manifest_key"),
        "queue_partition": manifest.get("queue_partition"),
        "checkpoint_count": _first_int(
            _count_from(manifest, "checkpoint_count"),
            _list_count(manifest, "checkpoint_refs"),
            _list_count(discovery, "checkpoint_refs"),
        ),
        "seen_checkpoint_count": _first_int(
            _count_from(manifest, "seen_checkpoint_count"),
            _list_count(manifest, "seen_checkpoint_refs"),
        ),
        "queued_checkpoint_count": _first_int(
            _count_from(manifest, "queued_checkpoint_count"),
            _list_count(manifest, "queued_checkpoint_refs"),
        ),
        "scan_spec": _mapping(manifest.get("scan_spec")),
        "rating_defaults": _mapping(manifest.get("rating_defaults")),
        "queue_len": _mapping(payload).get("queue_len") if isinstance(payload, Mapping) else None,
    }


def unwrap_rating_config(payload: Any) -> dict[str, Any]:
    data = _mapping(payload)
    nested = _mapping(data.get("rating_spec"))
    if nested:
        return nested
    return data


def summarize_rating_config(payload: Any | None) -> dict[str, Any]:
    if payload is None:
        return {"present": False}
    config = unwrap_rating_config(payload)
    checkpoints = _sequence(config.get("checkpoints"))
    if not checkpoints:
        checkpoints = _sequence(config.get("checkpoint_refs"))
    decision_source_frames = config.get("decision_source_frames")
    return {
        "present": True,
        "tournament_id": config.get("tournament_id"),
        "rating_run_id": config.get("rating_run_id"),
        "checkpoint_count": _first_int(_count_from(config, "checkpoint_count"), len(checkpoints)),
        "round_count": config.get("round_count"),
        "continue_from_latest": config.get("continue_from_latest"),
        "decision_source_frames": decision_source_frames,
        "one_frame_decisions": decision_source_frames == 1,
        "pair_selection": config.get("pair_selection"),
        "games_per_pair": config.get("games_per_pair"),
        "games_per_shard": config.get("games_per_shard"),
        "active_pool_limit": config.get("active_pool_limit"),
    }


def summarize_rating_claim(
    payload: Any | None,
    *,
    rating_claim_key: str | None = None,
) -> dict[str, Any]:
    if payload is None:
        return {"present": False, "rating_claim_key": rating_claim_key}
    claim = _mapping(payload)
    status = str(claim.get("status") or claim.get("previous_pointer_status") or "")
    missing = status in {"missing", "not_found", "none"}
    return {
        "present": not missing,
        "rating_claim_key": claim.get("rating_claim_key") or rating_claim_key,
        "schema_id": claim.get("schema_id"),
        "created_at": claim.get("created_at"),
        "stale_after_seconds": _count_from(claim, "stale_after_seconds"),
        "queue_len_before": _count_from(claim, "queue_len_before"),
        "queue_len_after_repair": _count_from(claim, "queue_len_after_repair"),
        "event_count": _count_from(claim, "event_count"),
        "repaired_stale_claim": claim.get("repaired_stale_claim"),
        "stale": claim.get("stale"),
        "status": claim.get("status"),
    }


def _latest_game_count(latest: Mapping[str, Any]) -> int | None:
    explicit = _count_from(latest, "game_count", "completed_game_count", "valid_game_count")
    if explicit is not None:
        return explicit
    pair_results = _sequence(latest.get("pair_rating_results"))
    if pair_results:
        total = 0
        saw_game_count = False
        for row in pair_results:
            item = _mapping(row)
            count = _first_int(
                _count_from(item, "valid_games", "requested_games"),
                _count_from(_mapping(item.get("tally")), "game_count", "completed_count"),
            )
            if count is not None:
                total += count
                saw_game_count = True
        if saw_game_count:
            return total
    pair_count = _count_from(latest, "pair_count")
    rating_spec = _mapping(latest.get("rating_spec"))
    games_per_pair = _count_from(rating_spec, "games_per_pair")
    if pair_count is not None and games_per_pair is not None:
        return pair_count * games_per_pair
    return None


def summarize_rating_latest(payload: Any | None) -> dict[str, Any]:
    if payload is None:
        return {"present": False}
    latest = _mapping(payload)
    ratings = _sequence(latest.get("ratings"))
    checkpoint_roster = _mapping(latest.get("checkpoint_roster"))
    status_counts: dict[str, int] = {}
    for row in ratings:
        status = str(_mapping(row).get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    active_count = status_counts.get("active", 0)
    return {
        "present": True,
        "tournament_id": latest.get("tournament_id"),
        "rating_run_id": latest.get("rating_run_id"),
        "round_id": latest.get("round_id"),
        "round_index": latest.get("round_index"),
        "checkpoint_count": _first_int(
            _count_from(latest, "checkpoint_count"),
            len(ratings) if ratings else None,
            len(checkpoint_roster) if checkpoint_roster else None,
        ),
        "rating_count": len(ratings),
        "active_count": active_count,
        "status_counts": status_counts,
        "pair_count": _count_from(latest, "pair_count"),
        "game_count": _latest_game_count(latest),
        "status": latest.get("status"),
        "phase": latest.get("phase"),
        "provisional": latest.get("provisional"),
    }


def summarize_progress_payload(payload: Any) -> dict[str, Any]:
    data = _mapping(payload)
    ticks = _sequence(data.get("ticks"))
    results = _sequence(data.get("results"))
    if ticks:
        return {
            "schema_id": data.get("schema_id"),
            "kind": "intake_tick_batch",
            "tick_count": _first_int(_count_from(data, "tick_count"), len(ticks)),
            "new_checkpoint_count": _count_from(data, "new_checkpoint_count"),
            "requested_key_count": _count_from(data, "requested_key_count"),
            "drained_manifest_count": None,
        }
    if results:
        return {
            "schema_id": data.get("schema_id"),
            "kind": "intake_drain_tick",
            "tick_count": None,
            "new_checkpoint_count": None,
            "event_count": _count_from(data, "event_count"),
            "active_manifest_count": _count_from(data, "active_manifest_count"),
            "drained_manifest_count": _count_from(data, "drained_manifest_count"),
            "rating_call_count": len(_sequence(data.get("rating_call_ids"))),
        }
    return {
        "schema_id": data.get("schema_id"),
        "kind": "progress",
        "tick_count": _count_from(data, "tick_count"),
        "new_checkpoint_count": _count_from(data, "new_checkpoint_count"),
        "event_count": _count_from(data, "event_count"),
        "rating_claimed": data.get("rating_claimed"),
        "rating_claim_stale": data.get("rating_claim_stale"),
        "rating_claim_repaired": data.get("rating_claim_repaired"),
        "rating_claim_key": data.get("rating_claim_key"),
        "spawn_skipped_reason": data.get("spawn_skipped_reason"),
        "queue_len_before": _count_from(data, "queue_len_before"),
        "queue_len_after_repair": _count_from(data, "queue_len_after_repair"),
        "status": data.get("status"),
        "phase": data.get("phase"),
    }


def build_summary(
    *,
    intake_manifest: Any | None = None,
    rating_config: Any | None = None,
    rating_latest: Any | None = None,
    rating_claim: Any | None = None,
    rating_claim_key: str | None = None,
    queue_len: int | None = None,
    progress_payloads: Sequence[Any] = (),
) -> dict[str, Any]:
    intake = summarize_intake_manifest(intake_manifest)
    config = summarize_rating_config(rating_config)
    latest = summarize_rating_latest(rating_latest)
    claim = summarize_rating_claim(rating_claim, rating_claim_key=rating_claim_key)
    if queue_len is None:
        queue_len = intake.get("queue_len") if isinstance(intake.get("queue_len"), int) else None
    if queue_len is not None and intake.get("present"):
        intake["queue_len"] = queue_len
    if (
        latest.get("present")
        and latest.get("game_count") is None
        and isinstance(latest.get("pair_count"), int)
    ):
        try:
            latest["game_count"] = int(latest["pair_count"]) * int(config["games_per_pair"])
        except (KeyError, TypeError, ValueError):
            pass
    tournament_id = (
        intake.get("tournament_id")
        or config.get("tournament_id")
        or latest.get("tournament_id")
    )
    rating_run_id = (
        intake.get("rating_run_id")
        or config.get("rating_run_id")
        or latest.get("rating_run_id")
    )
    summary = {
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "intake_manifest": intake,
        "rating_config": config,
        "rating_latest": latest,
        "rating_claim": claim,
        "queue": {
            "present": queue_len is not None,
            "queue_len": queue_len,
        },
        "progress": [summarize_progress_payload(payload) for payload in progress_payloads],
    }
    summary["warnings"] = warnings_for_summary(summary)
    return summary


def warnings_for_summary(summary: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    intake = _mapping(summary.get("intake_manifest"))
    config = _mapping(summary.get("rating_config"))
    latest = _mapping(summary.get("rating_latest"))
    claim = _mapping(summary.get("rating_claim"))
    queue = _mapping(summary.get("queue"))
    progress_rows = [_mapping(row) for row in _sequence(summary.get("progress"))]
    manifest_count = intake.get("checkpoint_count")
    seen_count = intake.get("seen_checkpoint_count")
    config_count = config.get("checkpoint_count")
    latest_count = latest.get("checkpoint_count")
    queued_count = intake.get("queued_checkpoint_count")
    queue_len = queue.get("queue_len")
    if queue_len is None:
        queue_len = intake.get("queue_len")
    fresh_claim_seen = bool(claim.get("present") and claim.get("stale") is not True)
    fresh_claim_seen = fresh_claim_seen or any(
        row.get("spawn_skipped_reason") == "rating_run_claim_exists"
        and row.get("rating_claim_stale") is False
        for row in progress_rows
    )

    if intake.get("present") and config.get("present") and isinstance(manifest_count, int):
        if isinstance(config_count, int) and manifest_count > config_count:
            warnings.append(
                "manifest sees "
                f"{manifest_count} checkpoints but rating config only has {config_count}"
            )
    if intake.get("present") and latest.get("present") and isinstance(manifest_count, int):
        if isinstance(latest_count, int) and manifest_count > latest_count:
            warnings.append(
                "manifest sees "
                f"{manifest_count} checkpoints but rating latest only has {latest_count}"
            )
    if intake.get("present") and not latest.get("present"):
        warnings.append("rating latest is missing")
    if config.get("present") and config.get("decision_source_frames") != 1:
        warnings.append(
            "rating config is not one-frame: "
            f"decision_source_frames={config.get('decision_source_frames')!r}"
        )
    rating_defaults = _mapping(intake.get("rating_defaults"))
    manifest_live = bool(intake.get("active", True))
    continue_from_latest = bool(
        config.get("continue_from_latest", rating_defaults.get("continue_from_latest", False))
    )
    if (
        manifest_live
        and intake.get("present")
        and config.get("present")
        and not continue_from_latest
        and isinstance(manifest_count, int)
        and isinstance(config_count, int)
        and manifest_count > config_count
    ):
        warnings.append("live intake appears to need continue_from_latest but it is false")
    if (
        isinstance(queued_count, int)
        and queued_count > 0
        and isinstance(config_count, int)
        and isinstance(manifest_count, int)
        and config_count < manifest_count
    ):
        warnings.append(
            f"intake queue has {queued_count} queued checkpoints but rating config "
            f"only covers {config_count} of {manifest_count}"
        )
    if queue_len == 0 and isinstance(queued_count, int) and queued_count > 0:
        warnings.append(
            f"queue_len=0 but intake manifest still has {queued_count} queued checkpoints"
        )
    if (
        queue_len == 0
        and fresh_claim_seen
        and isinstance(manifest_count, int)
        and isinstance(config_count, int)
        and manifest_count > config_count
    ):
        warnings.append(
            "recent failure shape detected: intake saw "
            f"{manifest_count}, rating config has {config_count}, queue_len=0, "
            "and a fresh rating claim exists"
        )
    if fresh_claim_seen:
        warnings.append("fresh rating claim exists; drain/spawn may be blocked")
    claim_event_count = claim.get("event_count")
    if (
        fresh_claim_seen
        and isinstance(claim_event_count, int)
        and isinstance(seen_count, int)
        and claim_event_count < seen_count
    ):
        warnings.append(
            "fresh rating claim event_count is smaller than manifest seen count: "
            f"{claim_event_count} < {seen_count}"
        )
    if (
        fresh_claim_seen
        and latest.get("present")
        and isinstance(manifest_count, int)
        and isinstance(latest_count, int)
        and latest_count < manifest_count
    ):
        warnings.append(
            "claim exists but rating latest is smaller than manifest: "
            f"{latest_count} < {manifest_count}"
        )
    if latest.get("present"):
        rating_count = latest.get("rating_count")
        active_count = latest.get("active_count")
        if rating_count == 0:
            warnings.append("rating latest has no rating rows")
        elif active_count == 0 and latest.get("status_counts"):
            warnings.append("rating latest has no active rows")
    return warnings


def _format_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return _yes_no(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _append_kv(lines: list[str], label: str, value: Any) -> None:
    lines.append(f"  {label}: {_format_value(value)}")


def format_text(summary: Mapping[str, Any], *, include_modal_hints: bool = False) -> str:
    lines: list[str] = ["CurvyTron tournament debug bundle"]
    _append_kv(lines, "tournament_id", summary.get("tournament_id"))
    _append_kv(lines, "rating_run_id", summary.get("rating_run_id"))

    intake = _mapping(summary.get("intake_manifest"))
    lines.append("")
    lines.append("Intake manifest")
    if not intake.get("present"):
        lines.append("  present: no")
    else:
        for key in (
            "checkpoint_count",
            "seen_checkpoint_count",
            "queued_checkpoint_count",
            "active",
            "queue_len",
            "manifest_key",
            "queue_partition",
        ):
            _append_kv(lines, key, intake.get(key))
        _append_kv(lines, "scan_spec", intake.get("scan_spec"))
        _append_kv(lines, "rating_defaults", intake.get("rating_defaults"))

    config = _mapping(summary.get("rating_config"))
    lines.append("")
    lines.append("Rating config")
    if not config.get("present"):
        lines.append("  present: no")
    else:
        for key in (
            "checkpoint_count",
            "round_count",
            "continue_from_latest",
            "decision_source_frames",
            "one_frame_decisions",
            "pair_selection",
            "games_per_pair",
            "games_per_shard",
            "active_pool_limit",
        ):
            _append_kv(lines, key, config.get(key))

    latest = _mapping(summary.get("rating_latest"))
    lines.append("")
    lines.append("Rating latest")
    if not latest.get("present"):
        lines.append("  present: no")
    else:
        for key in (
            "checkpoint_count",
            "rating_count",
            "active_count",
            "pair_count",
            "game_count",
            "status",
            "phase",
            "provisional",
            "round_id",
            "round_index",
            "status_counts",
        ):
            _append_kv(lines, key, latest.get(key))

    queue = _mapping(summary.get("queue"))
    if queue.get("present"):
        lines.append("")
        lines.append("Queue")
        _append_kv(lines, "queue_len", queue.get("queue_len"))

    claim = _mapping(summary.get("rating_claim"))
    if claim.get("present"):
        lines.append("")
        lines.append("Rating claim")
        for key in (
            "rating_claim_key",
            "schema_id",
            "created_at",
            "stale_after_seconds",
            "queue_len_before",
            "queue_len_after_repair",
            "event_count",
            "repaired_stale_claim",
            "stale",
            "status",
        ):
            _append_kv(lines, key, claim.get(key))

    progress_rows = _sequence(summary.get("progress"))
    if progress_rows:
        lines.append("")
        lines.append("Progress/ticks")
        for index, row in enumerate(progress_rows, start=1):
            payload = _mapping(row)
            lines.append(f"  payload_{index}:")
            for key in (
                "kind",
                "schema_id",
                "tick_count",
                "new_checkpoint_count",
                "event_count",
                "rating_claimed",
                "rating_claim_stale",
                "rating_claim_repaired",
                "rating_claim_key",
                "spawn_skipped_reason",
                "requested_key_count",
                "active_manifest_count",
                "drained_manifest_count",
                "rating_call_count",
                "queue_len_before",
                "queue_len_after_repair",
                "status",
                "phase",
            ):
                if key in payload and payload.get(key) is not None:
                    lines.append(f"    {key}: {_format_value(payload.get(key))}")

    warnings = _sequence(summary.get("warnings"))
    lines.append("")
    lines.append("Warnings")
    if warnings:
        for warning in warnings:
            lines.append(f"  WARNING: {warning}")
    else:
        lines.append("  none")

    if include_modal_hints:
        lines.append("")
        lines.extend(modal_hint_lines(summary))

    return "\n".join(lines)


def modal_hint_lines(
    summary: Mapping[str, Any],
    *,
    volume: str = DEFAULT_TOURNAMENT_VOLUME,
    output_dir: str = "/tmp/curvytron-debug",
) -> list[str]:
    tournament_id = summary.get("tournament_id")
    rating_run_id = summary.get("rating_run_id")
    if not tournament_id or not rating_run_id:
        return ["Modal hints", "  need tournament_id and rating_run_id"]
    base = f"tournaments/curvytron/{tournament_id}"
    manifest_key = intake_manifest_key(str(tournament_id), str(rating_run_id))
    default_claim_key = default_rating_claim_key(str(tournament_id), str(rating_run_id))
    intake = _mapping(summary.get("intake_manifest"))
    queue_partition = intake.get("queue_partition")
    if not queue_partition:
        queue_partition = intake_queue_partition(str(tournament_id), str(rating_run_id))
    refs = [
        (
            f"{base}/intake/{rating_run_id}/config.json",
            f"{output_dir}/intake_config.json",
        ),
        (
            f"{base}/intake/{rating_run_id}/progress.json",
            f"{output_dir}/intake_progress.json",
        ),
        (
            f"{base}/ratings/{rating_run_id}/config.json",
            f"{output_dir}/rating_config.json",
        ),
        (
            f"{base}/ratings/{rating_run_id}/latest.json",
            f"{output_dir}/rating_latest.json",
        ),
        (
            f"{base}/ratings/{rating_run_id}/progress.json",
            f"{output_dir}/rating_progress.json",
        ),
    ]
    lines = ["Modal hints", "  # volume artifacts", f"  mkdir -p {output_dir}"]
    for ref, target in refs:
        lines.append(f"  uv run --extra modal modal volume get {volume} {ref} {target}")
    lines.extend(
        [
            "  # Modal Dict / Queue state checks",
            f"  uv run --extra modal modal dict get {CHECKPOINT_INTAKE_DICT_NAME} {manifest_key}",
            f"  uv run --extra modal modal dict get {CHECKPOINT_INTAKE_DICT_NAME} {default_claim_key}",
            f"  uv run --extra modal modal queue len {CHECKPOINT_INTAKE_QUEUE_NAME} --partition {queue_partition}",
        ]
    )
    return lines


def _load_optional(path: str | None) -> Any | None:
    if not path:
        return None
    return _load_json_with_modal_noise(Path(path))


def _parse_queue_len_value(value: Any) -> int:
    if isinstance(value, Mapping):
        for key in ("queue_len", "len", "length", "count"):
            if key in value:
                return _parse_queue_len_value(value[key])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 1:
            return _parse_queue_len_value(value[0])
    text = str(value).strip()
    if text.startswith("{") or text.startswith("["):
        return _parse_queue_len_value(json.loads(text))
    match = re.search(r"-?\d+", text)
    if match is None:
        raise ValueError(f"could not parse queue length from {value!r}")
    return int(match.group(0))


def _load_queue_len_file(path: str | None) -> int | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        payload = _load_json_with_modal_noise(Path(path))
    except ValueError:
        payload = text
    return _parse_queue_len_value(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize local CurvyTron intake/rating JSON artifacts and flag "
            "obvious closed-loop mismatches."
        )
    )
    parser.add_argument(
        "--manifest",
        "--intake",
        "--intake-manifest",
        dest="intake_manifest",
        help="Local intake manifest JSON, or intake-status/seed JSON containing manifest.",
    )
    parser.add_argument("--rating-config", help="Local rating config JSON.")
    parser.add_argument("--rating-latest", help="Local rating latest JSON.")
    parser.add_argument(
        "--rating-claim",
        "--claim",
        dest="rating_claim",
        help="Local Modal Dict rating-claim JSON, if already fetched.",
    )
    parser.add_argument(
        "--claim-key",
        help="Rating claim key to print when the claim JSON does not include it.",
    )
    parser.add_argument("--queue-len", type=int, help="Local Modal Queue length value.")
    parser.add_argument(
        "--queue-len-file",
        help="Local file containing a queue length, or JSON with queue_len/len/count.",
    )
    parser.add_argument(
        "--intake-progress",
        "--intake-tick",
        "--rating-progress",
        dest="progress_paths",
        action="append",
        default=[],
        help="Local intake tick/drain/rating progress JSON. May be repeated.",
    )
    parser.add_argument("--tournament-id", help="Tournament id for hints if no JSON provides it.")
    parser.add_argument("--rating-run-id", help="Rating run id for hints if no JSON provides it.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON.")
    parser.add_argument(
        "--print-modal-hints",
        "--modal-hints",
        dest="print_modal_hints",
        action="store_true",
        help="Append Modal volume/Dict/Queue commands for expected state paths.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    progress_payloads = [_load_json_with_modal_noise(Path(path)) for path in args.progress_paths]
    queue_len = args.queue_len
    if queue_len is None:
        queue_len = _load_queue_len_file(args.queue_len_file)
    summary = build_summary(
        intake_manifest=_load_optional(args.intake_manifest),
        rating_config=_load_optional(args.rating_config),
        rating_latest=_load_optional(args.rating_latest),
        rating_claim=_load_optional(args.rating_claim),
        rating_claim_key=args.claim_key,
        queue_len=queue_len,
        progress_payloads=progress_payloads,
    )
    if args.tournament_id and not summary.get("tournament_id"):
        summary["tournament_id"] = args.tournament_id
    if args.rating_run_id and not summary.get("rating_run_id"):
        summary["rating_run_id"] = args.rating_run_id

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_text(summary, include_modal_hints=args.print_modal_hints))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
