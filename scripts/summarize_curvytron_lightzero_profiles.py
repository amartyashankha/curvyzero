#!/usr/bin/env python3
"""Summarize CurvyTron LightZero profile summary artifacts.

Examples:

    python3 scripts/summarize_curvytron_lightzero_profiles.py \
      --attempt opt-fixed-resource-smoke-s20260511i:fixed-gpu40cpu-c4-sim4-steps16

    python3 scripts/summarize_curvytron_lightzero_profiles.py \
      /private/tmp/fixed_gpu40cpu_c4_summary_s20260511i.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_VOLUME = "curvyzero-runs"
DEFAULT_TASK_ID = "lightzero-curvytron-visual-survival"
DEFAULT_DOWNLOAD_DIR = Path("/private/tmp/curvytron_lightzero_profile_summaries")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _get(mapping: dict[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _short_problem(summary: dict[str, Any]) -> str:
    problems = summary.get("problems")
    if not isinstance(problems, list) or not problems:
        return ""
    text = str(problems[0]).replace("\n", " ")
    return text[:140] + ("..." if len(text) > 140 else "")


def _safe_name(ref: str) -> str:
    digest = hashlib.sha256(ref.encode("utf-8")).hexdigest()[:12]
    stem = ref.strip("/").replace("/", "__").replace(":", "_")
    return f"{stem[:120]}__{digest}.json"


def _attempt_ref(task_id: str, attempt: str) -> str:
    if ":" not in attempt:
        raise ValueError(f"--attempt must be RUN_ID:ATTEMPT_ID, got {attempt!r}")
    run_id, attempt_id = attempt.split(":", 1)
    return f"training/{task_id}/{run_id}/attempts/{attempt_id}/train/summary.json"


def _resolve_input(
    source: str,
    *,
    volume: str,
    download_dir: Path,
    fetch: bool,
    wait_sec: float,
    poll_sec: float,
) -> Path:
    path = Path(source)
    if path.exists():
        return path
    ref = source.removeprefix("ref:")
    if not fetch:
        raise FileNotFoundError(f"{source!r} is not a local file and --no-fetch was set")
    download_dir.mkdir(parents=True, exist_ok=True)
    dest = download_dir / _safe_name(ref)
    if not dest.exists():
        deadline = time.monotonic() + max(0.0, wait_sec)
        while True:
            result = subprocess.run(
                ["modal", "volume", "get", volume, ref, str(dest)],
                check=False,
            )
            if result.returncode == 0:
                break
            if time.monotonic() >= deadline:
                result.check_returncode()
            time.sleep(max(0.25, poll_sec))
    return dest


def summarize_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    phase = _as_dict(summary.get("phase_profile"))
    command = _as_dict(summary.get("command"))
    auto_resume = _as_dict(command.get("auto_resume"))
    counts = _as_dict(phase.get("counts"))
    timers = _as_dict(phase.get("timers_sec"))
    derived = _as_dict(phase.get("derived_stats"))
    action = _as_dict(summary.get("action_observability"))
    gpu = _as_dict(phase.get("gpu_sampling"))
    runtime = _as_dict(summary.get("runtime_compute"))

    wall = _float(timers.get("train_muzero_wall_sec"))
    steps = _int(counts.get("env_steps_collected"))
    steps_per_sec = (steps / wall) if steps is not None and wall else None
    telemetry_sec = _float(timers.get("env_telemetry_write_sec"))
    telemetry_pct = (telemetry_sec / wall * 100.0) if telemetry_sec is not None and wall else None
    mcts_sec = _float(timers.get("mcts_search_sec"))
    mcts_sim_budget = _int(counts.get("mcts_search_simulation_budget_sum"))
    root_sum = _int(counts.get("mcts_search_root_sum"))
    search_calls = _int(counts.get("mcts_search_calls"))
    sims_per_root = _int(command.get("num_simulations"))
    root_sim_budget = root_sum * sims_per_root if root_sum is not None and sims_per_root else None
    root_sims_per_sec = (
        root_sim_budget / mcts_sec if root_sim_budget is not None and mcts_sec else None
    )
    mcts_sims_per_sec = (
        mcts_sim_budget / mcts_sec if mcts_sim_budget is not None and mcts_sec else None
    )
    root_batch_mean = derived.get("mcts_search_root_batch_mean")
    if root_batch_mean is None and root_sum is not None and search_calls:
        root_batch_mean = root_sum / search_calls

    return {
        "path": str(path),
        "ok": summary.get("ok"),
        "status": summary.get("status"),
        "run_id": summary.get("run_id"),
        "attempt_id": summary.get("attempt_id"),
        "compute": summary.get("compute"),
        "env_variant": command.get("env_variant"),
        "reward_variant": command.get("reward_variant"),
        "manager": command.get("env_manager_type"),
        "collectors": command.get("collector_env_num"),
        "episodes": command.get("n_episode"),
        "sims": command.get("num_simulations"),
        "batch": command.get("batch_size"),
        "lightzero_multi_gpu": command.get("lightzero_multi_gpu"),
        "profile_cuda_sync_enabled": command.get("profile_cuda_sync_enabled"),
        "profile_allow_auto_resume": command.get("profile_allow_auto_resume"),
        "auto_resume_found": auto_resume.get("found"),
        "source_max_steps": command.get("source_max_steps"),
        "telemetry_stride": command.get("env_telemetry_stride"),
        "steps": steps,
        "wall_sec": wall,
        "steps_per_sec": steps_per_sec,
        "mcts_sims_per_sec": mcts_sims_per_sec,
        "root_sims_per_sec": root_sims_per_sec,
        "collector_sec": _float(timers.get("collector_collect_sec")),
        "mcts_sec": mcts_sec,
        "policy_collect_sec": _float(timers.get("policy_forward_collect_sec")),
        "policy_eval_sec": _float(timers.get("policy_forward_eval_sec")),
        "model_initial_sec": _float(timers.get("model_initial_inference_sec")),
        "model_recurrent_sec": _float(timers.get("model_recurrent_inference_sec")),
        "learner_sec": _float(timers.get("learner_train_sec")),
        "replay_sec": _float(timers.get("replay_sample_sec")),
        "eval_sec": _float(timers.get("evaluator_eval_sec")),
        "checkpoint_sec": _float(timers.get("learner_save_checkpoint_sec")),
        "cuda_sync_sec": _float(timers.get("cuda_sync_sec")),
        "telemetry_sec": telemetry_sec,
        "telemetry_pct": telemetry_pct,
        "telemetry_rows": action.get("row_count"),
        "telemetry_scope": action.get("counts_scope"),
        "root_batch_mean": root_batch_mean,
        "recurrent_batch_mean": derived.get(
            "model_recurrent_inference_in_mcts_search_batch_mean"
        ),
        "gpu_max_pct": gpu.get("max_gpu_util_percent"),
        "gpu_mem_mib": gpu.get("max_memory_used_mib"),
        "cuda_available": runtime.get("torch_cuda_available"),
        "cuda_device_count": runtime.get("torch_cuda_device_count"),
        "problem": _short_problem(summary),
    }


TABLE_COLUMNS = [
    ("ok", "ok"),
    ("compute", "compute"),
    ("attempt_id", "attempt"),
    ("collectors", "c"),
    ("episodes", "ep"),
    ("sims", "sim"),
    ("source_max_steps", "src"),
    ("lightzero_multi_gpu", "lz_mgpu"),
    ("profile_cuda_sync_enabled", "cuda_sync"),
    ("auto_resume_found", "resume"),
    ("cuda_device_count", "cuda_n"),
    ("steps", "steps"),
    ("wall_sec", "wall"),
    ("steps_per_sec", "steps/s"),
    ("collector_sec", "collect"),
    ("mcts_sec", "mcts"),
    ("root_sims_per_sec", "root_sims/s"),
    ("policy_collect_sec", "p_collect"),
    ("learner_sec", "learner"),
    ("replay_sec", "replay"),
    ("eval_sec", "eval"),
    ("cuda_sync_sec", "cuda_sync_s"),
    ("telemetry_stride", "tel_stride"),
    ("telemetry_rows", "tel_rows"),
    ("telemetry_pct", "tel_%"),
    ("gpu_max_pct", "gpu_%"),
    ("problem", "problem"),
]


def _render_table(rows: list[dict[str, Any]]) -> str:
    rendered: list[list[str]] = []
    headers = [header for _, header in TABLE_COLUMNS]
    rendered.append(headers)
    for row in rows:
        rendered.append(
            [
                _fmt(row.get(key), digits=2)
                if key not in {"problem", "attempt_id", "compute"}
                else str(row.get(key) or "")
                for key, _ in TABLE_COLUMNS
            ]
        )
    widths = [max(len(line[i]) for line in rendered) for i in range(len(headers))]
    lines = []
    for index, line in enumerate(rendered):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)))
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="local summary.json paths or Modal refs")
    parser.add_argument(
        "--attempt",
        action="append",
        default=[],
        metavar="RUN_ID:ATTEMPT_ID",
        help="summarize training/<task>/RUN_ID/attempts/ATTEMPT_ID/train/summary.json",
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument(
        "--wait-sec",
        type=float,
        default=0.0,
        help="wait up to this many seconds for Modal refs that are not ready yet",
    )
    parser.add_argument("--poll-sec", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", help="emit JSON rows instead of a table")
    args = parser.parse_args()

    sources = list(args.inputs)
    sources.extend(_attempt_ref(args.task_id, attempt) for attempt in args.attempt)
    if not sources:
        parser.error("provide at least one summary path/ref or --attempt RUN:ATTEMPT")

    paths = [
        _resolve_input(
            source,
            volume=args.volume,
            download_dir=args.download_dir,
            fetch=not args.no_fetch,
            wait_sec=args.wait_sec,
            poll_sec=args.poll_sec,
        )
        for source in sources
    ]
    rows = [summarize_summary(path) for path in paths]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(_render_table(rows))


if __name__ == "__main__":
    main()
