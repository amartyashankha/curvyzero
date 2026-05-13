#!/usr/bin/env python3
"""Summarize local CurvyTron optimizer profile result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path(
    "artifacts/local/curvytron_optimizer_profile_results/"
    "opt-stock-frozen-profile-first-wave-20260512e"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _round(value: Any, digits: int = 2) -> Any:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, digits)
    return value


def _profile_row(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    compact = payload.get("compact") or {}
    command = compact.get("command") or {}
    counts = compact.get("counts") or {}
    derived = compact.get("derived") or {}
    timers = compact.get("timers_sec") or {}
    gpu = compact.get("gpu") or {}
    telemetry = compact.get("telemetry") or {}
    env_timing = telemetry.get("profile_env_timing_sec") or {}
    sampled_sum = env_timing.get("sampled_sum") or {}
    return {
        "row": str(payload.get("row_id") or path.stem.removeprefix("row_").removesuffix("_result")),
        "status": payload.get("status"),
        "collectors": command.get("collector_env_num"),
        "sims": command.get("num_simulations"),
        "death": "nodeath" if command.get("disable_death_for_profile") else "normal",
        "render": command.get("source_state_trail_render_mode"),
        "reward": command.get("reward_variant"),
        "steps": counts.get("env_steps_collected"),
        "wall": timers.get("train_muzero_wall"),
        "steps_s": derived.get("steps_per_sec"),
        "collect": timers.get("collector_collect"),
        "mcts": timers.get("mcts_search"),
        "learner": timers.get("learner_train"),
        "policy_collect": timers.get("policy_forward_collect"),
        "obs_sum": sampled_sum.get("observation_sec"),
        "opp_sum": sampled_sum.get("opponent_action_sec"),
        "vec_sum": sampled_sum.get("vector_step_sec"),
        "gpu_max": gpu.get("max_util_percent"),
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("row", "row"),
        ("collectors", "C"),
        ("sims", "sims"),
        ("death", "death"),
        ("render", "render"),
        ("reward", "reward"),
        ("steps", "steps"),
        ("wall", "wall"),
        ("steps_s", "steps/s"),
        ("collect", "collect"),
        ("mcts", "MCTS"),
        ("learner", "learner"),
        ("obs_sum", "obs"),
        ("opp_sum", "opp"),
        ("gpu_max", "GPU max"),
    ]
    header = "| " + " | ".join(label for _key, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in sorted(rows, key=lambda item: str(item["row"])):
        cells = []
        for key, _label in columns:
            value = _round(row.get(key))
            cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.results_dir.glob("row_*_result.json"))
    rows = [_profile_row(path) for path in paths]
    if args.format == "json":
        text = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    else:
        text = _markdown_table(rows) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
