"""Inspect existing dummy Pong replay and observability artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

KNOWN_ARTIFACTS = (
    "summary.json",
    "replay_rows.jsonl",
    "frames.jsonl",
    "games.jsonl",
    "steps.jsonl",
)


def inspect_dummy_pong_artifacts(
    artifact_dir: Path,
    *,
    sample_frames: int = 3,
) -> dict[str, Any]:
    """Return a compact quality summary for a dummy Pong artifact directory."""

    artifact_dir = Path(artifact_dir)
    if sample_frames < 0:
        raise ValueError("sample_frames must be non-negative")

    detected_files = _detected_files(artifact_dir)
    summary = _load_json(artifact_dir / "summary.json") if "summary.json" in detected_files else None
    replay = _inspect_replay_rows(artifact_dir / "replay_rows.jsonl")
    games = _inspect_games(artifact_dir / "games.jsonl")
    steps = _inspect_steps(artifact_dir / "steps.jsonl")
    frames = _inspect_frames(artifact_dir / "frames.jsonl", sample_frames=sample_frames)

    row_counts = {
        "replay_rows": replay["row_count"],
        "games": games["row_count"],
        "steps": steps["row_count"],
        "frames": frames["row_count"],
    }
    row_counts = {name: count for name, count in row_counts.items() if count is not None}

    raster = _merge_raster_summaries(summary, replay, frames)
    terminal_counts = {
        "replay_rows": replay["terminal_counts"],
        "games": games["terminal_counts"],
        "steps": steps["terminal_counts"],
        "frames": frames["terminal_counts"],
    }
    terminal_counts = {
        name: counts
        for name, counts in terminal_counts.items()
        if counts["terminated"] or counts["truncated"]
    }

    result: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "detected_files": detected_files,
        "row_counts": row_counts,
        "schemas": _schema_summary(summary, replay, games, steps, frames),
        "raster": raster,
        "action_histograms": replay["action_histograms"],
        "reward_totals": {
            "replay_rows": replay["reward_totals"],
            "games": games["reward_totals"],
            "steps": steps["reward_totals"],
        },
        "reward_row_counts": {
            "replay_rows": replay["reward_row_counts"],
        },
        "terminal_counts": terminal_counts,
        "sample_frame_strings": frames["sample_frame_strings"],
        "quality_notes": _quality_notes(summary, replay, games, steps, frames),
    }
    return result


def _detected_files(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    detected = {}
    for name in KNOWN_ARTIFACTS:
        path = artifact_dir / name
        if path.exists():
            detected[name] = {"bytes": path.stat().st_size}
    return detected


def _load_json(path: Path) -> dict[str, Any] | None:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield payload


def _inspect_replay_rows(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_file_summary()

    row_count = 0
    schemas = Counter()
    action_by_agent: dict[str, Counter[str]] = {}
    reward_totals: Counter[str] = Counter()
    reward_value_counts: Counter[str] = Counter()
    reward_row_counts = Counter()
    terminal_counts = Counter()
    raster_shapes = Counter()
    first_raster_schema = None

    for row in _read_jsonl(path):
        row_count += 1
        schemas.update([str(row.get("schema_id"))])
        agent = str(row.get("ego_agent", "unknown"))
        label = str(row.get("target_action_label", row.get("target_action_id", "unknown")))
        action_by_agent.setdefault(agent, Counter()).update([label])
        reward = float(row.get("reward_after_step", 0.0))
        reward_totals[agent] += reward
        reward_value_counts[f"{reward:g}"] += 1
        if reward > 0.0:
            reward_row_counts["positive"] += 1
        elif reward < 0.0:
            reward_row_counts["negative"] += 1
        else:
            reward_row_counts["zero"] += 1
        terminal_counts["terminated"] += int(bool(row.get("terminated_after_step")))
        terminal_counts["truncated"] += int(bool(row.get("truncated_after_step")))
        shape = _shape_from_grid(row.get("raster_grid")) or _tuple_shape(row.get("raster_shape"))
        if shape is not None:
            raster_shapes.update([shape])
        if first_raster_schema is None:
            first_raster_schema = row.get("raster_observation_schema_id")

    return {
        "row_count": row_count,
        "schemas": _counter_dict(schemas),
        "action_histograms": {
            agent: _counter_dict(counter)
            for agent, counter in sorted(action_by_agent.items())
        },
        "reward_totals": _float_counter_dict(reward_totals),
        "reward_row_counts": {
            "positive": int(reward_row_counts["positive"]),
            "negative": int(reward_row_counts["negative"]),
            "zero": int(reward_row_counts["zero"]),
            "nonzero": int(reward_row_counts["positive"] + reward_row_counts["negative"]),
            "values": _counter_dict(reward_value_counts),
        },
        "terminal_counts": _terminal_dict(terminal_counts),
        "raster_shapes": _shape_counter_dict(raster_shapes),
        "raster_schema_id": first_raster_schema,
    }


def _inspect_games(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_file_summary()

    row_count = 0
    schemas = Counter()
    reward_totals: Counter[str] = Counter()
    terminal_counts = Counter()

    for row in _read_jsonl(path):
        row_count += 1
        schemas.update([str(row.get("schema_id"))])
        terminal_counts["terminated"] += int(bool(row.get("terminated")))
        terminal_counts["truncated"] += int(bool(row.get("truncated")))
        for agent, reward in _dict_items(row.get("total_rewards")):
            reward_totals[str(agent)] += float(reward)

    return {
        "row_count": row_count,
        "schemas": _counter_dict(schemas),
        "reward_totals": _float_counter_dict(reward_totals),
        "terminal_counts": _terminal_dict(terminal_counts),
        "raster_shapes": {},
        "raster_schema_id": None,
    }


def _inspect_steps(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_file_summary()

    row_count = 0
    schemas = Counter()
    reward_totals: Counter[str] = Counter()
    terminal_counts = Counter()

    for row in _read_jsonl(path):
        row_count += 1
        schemas.update([str(row.get("schema_id"))])
        terminal_counts["terminated"] += int(bool(row.get("terminated")))
        terminal_counts["truncated"] += int(bool(row.get("truncated")))
        for agent, reward in _dict_items(row.get("rewards")):
            reward_totals[str(agent)] += float(reward)

    return {
        "row_count": row_count,
        "schemas": _counter_dict(schemas),
        "reward_totals": _float_counter_dict(reward_totals),
        "terminal_counts": _terminal_dict(terminal_counts),
        "raster_shapes": {},
        "raster_schema_id": None,
    }


def _inspect_frames(path: Path, *, sample_frames: int) -> dict[str, Any]:
    if not path.exists():
        return _empty_file_summary() | {"sample_frame_strings": []}

    row_count = 0
    schemas = Counter()
    terminal_counts = Counter()
    raster_shapes = Counter()
    sample_frame_strings: list[dict[str, Any]] = []
    first_raster_schema = None

    for row in _read_jsonl(path):
        row_count += 1
        schemas.update([str(row.get("schema_id"))])
        terminal_counts["terminated"] += int(bool(row.get("terminated")))
        terminal_counts["truncated"] += int(bool(row.get("truncated")))
        shape = _shape_from_grid(row.get("grid")) or _tuple_shape(row.get("shape"))
        if shape is not None:
            raster_shapes.update([shape])
        if first_raster_schema is None:
            first_raster_schema = row.get("raster_observation_schema_id")
        if len(sample_frame_strings) < sample_frames:
            sample_frame_strings.append(
                {
                    "frame_id": row.get("frame_id"),
                    "step_index": row.get("step_index"),
                    "grid": row.get("grid"),
                }
            )

    return {
        "row_count": row_count,
        "schemas": _counter_dict(schemas),
        "reward_totals": {},
        "terminal_counts": _terminal_dict(terminal_counts),
        "raster_shapes": _shape_counter_dict(raster_shapes),
        "raster_schema_id": first_raster_schema,
        "sample_frame_strings": sample_frame_strings,
    }


def _empty_file_summary() -> dict[str, Any]:
    return {
        "row_count": None,
        "schemas": {},
        "action_histograms": {},
        "reward_totals": {},
        "reward_row_counts": {
            "positive": 0,
            "negative": 0,
            "zero": 0,
            "nonzero": 0,
            "values": {},
        },
        "terminal_counts": {"terminated": 0, "truncated": 0},
        "raster_shapes": {},
        "raster_schema_id": None,
    }


def _merge_raster_summaries(
    summary: dict[str, Any] | None,
    replay: dict[str, Any],
    frames: dict[str, Any],
) -> dict[str, Any]:
    summary_shape = None
    encoding = None
    legend = None
    if summary:
        raster = summary.get("raster")
        if isinstance(raster, dict):
            summary_shape = _tuple_shape(raster.get("shape"))
            encoding = raster.get("encoding")
            legend = raster.get("legend")

    shapes = Counter()
    for encoded, count in replay["raster_shapes"].items():
        shapes.update({_decode_shape(encoded): count})
    for encoded, count in frames["raster_shapes"].items():
        shapes.update({_decode_shape(encoded): count})

    payload = {
        "observed_shapes": _shape_counter_dict(shapes),
        "schema_ids": sorted(
            {
                schema
                for schema in (replay["raster_schema_id"], frames["raster_schema_id"])
                if schema
            }
        ),
        "encoding": encoding,
        "legend": legend,
    }
    if summary_shape is not None:
        payload["summary_shape"] = f"{summary_shape[0]}x{summary_shape[1]}"
    return payload


def _schema_summary(
    summary: dict[str, Any] | None,
    replay: dict[str, Any],
    games: dict[str, Any],
    steps: dict[str, Any],
    frames: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary_schema_id": summary.get("summary_schema_id") if summary else None,
        "replay_rows": replay["schemas"],
        "games": games["schemas"],
        "steps": steps["schemas"],
        "frames": frames["schemas"],
    }
    return {key: value for key, value in payload.items() if value}


def _quality_notes(
    summary: dict[str, Any] | None,
    replay: dict[str, Any],
    games: dict[str, Any],
    steps: dict[str, Any],
    frames: dict[str, Any],
) -> list[str]:
    notes = []
    if summary and replay["row_count"] is not None:
        expected_rows = summary.get("total_rows")
        if expected_rows is not None and int(expected_rows) != replay["row_count"]:
            notes.append(f"summary total_rows={expected_rows} but replay_rows has {replay['row_count']}")
    if summary and steps["row_count"] is not None:
        expected_steps = summary.get("total_step_rows")
        if expected_steps is not None and int(expected_steps) != steps["row_count"]:
            notes.append(f"summary total_step_rows={expected_steps} but steps has {steps['row_count']}")
    if summary and frames["row_count"] is not None:
        expected_frames = summary.get("total_frame_rows")
        if expected_frames is not None and int(expected_frames) != frames["row_count"]:
            notes.append(f"summary total_frame_rows={expected_frames} but frames has {frames['row_count']}")
    if games["row_count"] and frames["row_count"] and steps["row_count"]:
        minimum_expected_frames = steps["row_count"] + games["row_count"]
        if frames["row_count"] < minimum_expected_frames:
            notes.append(
                "frames has fewer rows than steps plus reset frames "
                f"({frames['row_count']} < {minimum_expected_frames})"
            )
    if replay["row_count"]:
        replay_reward_counts = replay["reward_row_counts"]
        if replay_reward_counts["nonzero"] == 0:
            notes.append(
                "replay rewards are all zero; useful for imitation/action debugging, "
                "not reward-learning evidence"
            )
        elif replay_reward_counts["positive"] and replay_reward_counts["negative"]:
            notes.append("replay has both positive and negative reward rows")
        if replay["terminal_counts"]["truncated"] and not replay["terminal_counts"]["terminated"]:
            notes.append("replay terminal flags are truncations only")
    if summary:
        outcome_summary = summary.get("outcome_summary")
        if isinstance(outcome_summary, dict) and int(outcome_summary.get("truncations", 0)) == int(
            summary.get("games", -1)
        ):
            notes.append("summary reports every replay game truncated at the max-step cap")
    if len(replay["raster_shapes"]) + len(frames["raster_shapes"]) == 0:
        notes.append("no raster grids found in replay_rows or frames")
    if not notes:
        notes.append("no obvious count or raster-shape problems detected")
    return notes


def _shape_from_grid(grid: object) -> tuple[int, int] | None:
    if not isinstance(grid, list) or not grid:
        return None
    first = grid[0]
    if not isinstance(first, str):
        return None
    return (len(grid), len(first))


def _tuple_shape(value: object) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    return (int(value[0]), int(value[1]))


def _shape_counter_dict(counter: Counter[tuple[int, int]]) -> dict[str, int]:
    return {f"{height}x{width}": int(count) for (height, width), count in sorted(counter.items())}


def _decode_shape(encoded: str) -> tuple[int, int]:
    height, width = encoded.split("x", maxsplit=1)
    return int(height), int(width)


def _terminal_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        "terminated": int(counter["terminated"]),
        "truncated": int(counter["truncated"]),
    }


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _float_counter_dict(counter: Counter[str]) -> dict[str, float]:
    return {str(key): float(value) for key, value in sorted(counter.items())}


def _dict_items(value: object) -> Iterable[tuple[object, object]]:
    if isinstance(value, dict):
        return value.items()
    return ()
