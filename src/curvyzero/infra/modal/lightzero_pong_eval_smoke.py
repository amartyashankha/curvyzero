"""Tiny eval-only smoke for a loaded stock LightZero visual Atari Pong checkpoint.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke

This is intentionally mechanical infrastructure. It loads the mirrored
``iteration_1.pth.tar`` checkpoint, creates a real ALE-backed
``PongNoFrameskip-v4`` LightZero env with the same tiny visual config, and
runs only a few capped eval steps. It is not a quality or score run.
"""

from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import math
import os
import random
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_atari_rom_image import build_lightzero_atari_rom_image
from curvyzero.infra.modal.lightzero_pong_checkpoint_probe import (
    DEFAULT_ATTEMPT_ID,
    DEFAULT_CHECKPOINT_REF,
    DEFAULT_RUN_ID,
    _checkpoint_summary,
    _find_state_dict,
    _load_state_dict_probe,
    _summarize_value,
    _torch_load,
    _to_plain,
)
from curvyzero.infra.modal.lightzero_pong_dry_config_smoke import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTOR_ENV_NUM,
    DEFAULT_ENV_ID,
    DEFAULT_EVALUATOR_ENV_NUM,
    DEFAULT_MAX_ENV_STEP,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_SEED,
    DEFAULT_UPDATE_PER_COLLECT,
    LIGHTZERO_VERSION,
)
from curvyzero.infra.modal.lightzero_pong_tiny_train_smoke import (
    CHEAP_GPU_RESOURCE,
    DEFAULT_GAME_SEGMENT_LENGTH,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_MAX_TRAIN_ITER,
    TASK_ID,
    VOLUME_NAME,
    _patched_stock_atari_pong_configs,
    _runtime_compute_summary,
    _version_or_missing,
)


APP_NAME = "curvyzero-lightzero-pong-eval-smoke"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_MAX_EVAL_STEPS = 8
DEFAULT_ALLOW_MODEL_FALLBACK = True
DEFAULT_STEP_DETAIL_LIMIT = 32
DEFAULT_RUN_STOCK_EVALUATOR = False
DEFAULT_SKIP_MANUAL_ROLLOUT = False
DEFAULT_SUMMARY_ONLY = False
DEFAULT_QUIET_FRAMEWORK_LOGS = True
DEFAULT_OPTIMIZER_PHASE_TIMING = False
EVAL_FUNCTION_TIMEOUT_SEC = 16 * 60
DEFAULT_PARALLEL_EVAL_ID = "checkpoint_curve"
DEFAULT_EVAL_SEED_COUNT = 8
DEFAULT_EVAL_SEED_RNG_SEED = 20260510
EVAL_SEED_MIN = 0
EVAL_SEED_MAX = (2**31) - 1
DEFAULT_LOW_DETAIL_MAX_EVAL_STEPS = 64
DEFAULT_LOW_DETAIL_STEP_DETAIL_LIMIT = 2
DEFAULT_HIGH_DETAIL_MAX_EVAL_STEPS = 512
DEFAULT_HIGH_DETAIL_STEP_DETAIL_LIMIT = 8
GPU_L4_T4_CPU8_COMPUTE = "gpu-l4-t4-cpu8"
GPU_L4_T4_CPU40_COMPUTE = "gpu-l4-t4-cpu40"
DEFAULT_COMPUTE = GPU_L4_T4_CPU40_COMPUTE
DEFAULT_PARALLEL_RETENTION_POLICY = (
    "Keep all per-checkpoint eval JSON artifacts and the manifest until the "
    "run is manually archived; artifacts are intentionally named by checkpoint, "
    "detail pass, step cap, seed, and UTC stamp so low-detail and high-detail "
    "reruns can coexist."
)
STOCK_EVALUATOR_PATH = "lzero.worker.MuZeroEvaluator"
GENERIC_EVALUATOR_PATH = "ding.worker.InteractionSerialEvaluator"

image = (
    build_lightzero_atari_rom_image(lightzero_version=LIGHTZERO_VERSION)
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)


def _seed_panel_label(seeds: list[int]) -> str:
    payload = ",".join(str(seed) for seed in seeds)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"n{len(seeds)}_{digest}"
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-16:],
    }


@contextlib.contextmanager
def _quiet_framework_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


@contextlib.contextmanager
def _optimizer_timed_phase(timers: dict[str, float] | None, name: str):
    if timers is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        timers[name] = round(time.perf_counter() - started, 6)


def _default_output_ref(*, run_id: str, attempt_id: str) -> str:
    return (
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, "iteration_1_tiny")
        / f"lightzero_visual_pong_eval_smoke_{runs.utc_stamp()}.json"
    ).as_posix()


def _safe_generated_id(raw: str, *, fallback: str) -> str:
    cleaned = "".join(
        char if char in runs.SAFE_ID_CHARS else "_" for char in raw
    ).strip("._-")
    if not cleaned:
        cleaned = fallback
    if not cleaned[0].isalnum():
        cleaned = f"{fallback}_{cleaned}"
    return runs.clean_id(cleaned[:80], label=fallback)


def _checkpoint_label(checkpoint_ref: str, *, index: int) -> str:
    name = Path(checkpoint_ref).name
    for suffix in (".pth.tar", ".tar", ".pth", ".pt", ".bin"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name:
        name = f"checkpoint_{index:03d}"
    return _safe_generated_id(name, fallback=f"checkpoint_{index:03d}")


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_iterations(value: str | None) -> list[int]:
    iterations = []
    for item in _split_csv(value):
        try:
            iteration = int(item)
        except ValueError as exc:
            raise ValueError(f"selected iteration must be an int, got {item!r}") from exc
        if iteration < 0:
            raise ValueError("selected iterations must be non-negative")
        iterations.append(iteration)
    return iterations


def _parse_eval_seeds(
    *,
    seed: int,
    eval_seeds: str | None,
    eval_seed_count: int | None,
    eval_seed_rng_seed: int,
) -> tuple[list[int], int | None]:
    if eval_seeds is None:
        if eval_seed_count is None:
            return [seed], None
        if eval_seed_count < 1:
            raise ValueError("eval_seed_count must be >= 1")
        rng = random.Random(eval_seed_rng_seed)
        return (
            rng.sample(range(EVAL_SEED_MIN, EVAL_SEED_MAX + 1), eval_seed_count),
            eval_seed_rng_seed,
        )
    if eval_seed_count is not None:
        raise ValueError("eval_seeds cannot be combined with eval_seed_count")
    parsed: list[int] = []
    for item in _split_csv(eval_seeds):
        try:
            parsed_seed = int(item)
        except ValueError as exc:
            raise ValueError(f"eval seed must be an int, got {item!r}") from exc
        if parsed_seed not in parsed:
            parsed.append(parsed_seed)
    if not parsed:
        raise ValueError("eval_seeds must include at least one integer")
    return parsed, None


def _infer_checkpoint_template(checkpoint_ref: str) -> str:
    template = re.sub(
        r"iteration_\d+(\.pth\.tar|\.tar|\.pth|\.pt|\.bin)$",
        r"iteration_{iteration}\1",
        checkpoint_ref,
    )
    if template == checkpoint_ref:
        raise ValueError(
            "selected_iterations requires checkpoint_ref_template when checkpoint_ref "
            "does not end with iteration_<n>.pth.tar"
        )
    return template


def _selected_checkpoint_refs(
    *,
    checkpoint_ref: str,
    checkpoint_refs: str | None,
    checkpoint_ref_template: str | None,
    selected_iterations: str | None,
) -> list[str]:
    explicit_refs = _split_csv(checkpoint_refs)
    iterations = _parse_iterations(selected_iterations)
    if explicit_refs and (checkpoint_ref_template or iterations):
        raise ValueError("use either checkpoint_refs or a checkpoint template/selected_iterations")
    if explicit_refs:
        return explicit_refs
    if iterations:
        template = checkpoint_ref_template or _infer_checkpoint_template(checkpoint_ref)
        return [template.format(iteration=iteration) for iteration in iterations]
    if checkpoint_ref_template:
        raise ValueError("checkpoint_ref_template requires selected_iterations")
    return [checkpoint_ref]


def _detail_settings(
    *,
    eval_pass: str,
    max_eval_steps: int,
    step_detail_limit: int | None,
    low_detail_max_eval_steps: int,
    low_detail_step_detail_limit: int,
    high_detail_max_eval_steps: int,
    high_detail_step_detail_limit: int,
) -> tuple[int, int | None]:
    if eval_pass == "custom":
        return max_eval_steps, step_detail_limit
    if eval_pass == "low":
        return low_detail_max_eval_steps, low_detail_step_detail_limit
    if eval_pass == "high":
        return high_detail_max_eval_steps, high_detail_step_detail_limit
    raise ValueError("eval_pass must be one of: custom, low, high")


def _eval_cap_warning(*, max_episode_steps: int, max_eval_steps: int) -> dict[str, Any] | None:
    if max_episode_steps == max_eval_steps:
        return None
    return {
        "kind": "eval_cap_mismatch",
        "message": (
            "Manual eval uses max_eval_steps, but the LightZero env episode cap "
            "uses max_episode_steps. Set --max-episode-steps to the same value "
            "as the eval step cap when comparing manual and stock evaluator output."
        ),
        "max_episode_steps": max_episode_steps,
        "max_eval_steps": max_eval_steps,
    }


def _parallel_output_ref(
    *,
    run_id: str,
    attempt_id: str,
    eval_id: str,
    checkpoint_label: str,
    eval_pass: str,
    max_eval_steps: int,
    seed: int,
    stamp: str,
) -> str:
    leaf = (
        f"lightzero_visual_pong_eval_{checkpoint_label}_{eval_pass}"
        f"_steps{max_eval_steps}_seed{seed}_{stamp}.json"
    )
    return (
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, eval_id)
        / f"{checkpoint_label}_{eval_pass}_steps{max_eval_steps}_seed{seed}"
        / leaf
    ).as_posix()


def _parallel_table_row(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    episode = result.get("episode") if isinstance(result.get("episode"), dict) else {}
    status = result.get("status") if isinstance(result.get("status"), dict) else {}
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    manual_vs_stock = (
        result.get("manual_vs_stock") if isinstance(result.get("manual_vs_stock"), dict) else {}
    )
    stock_evaluator = (
        result.get("stock_evaluator") if isinstance(result.get("stock_evaluator"), dict) else {}
    )
    stock_rollout = (
        stock_evaluator.get("stock_rollout")
        if isinstance(stock_evaluator.get("stock_rollout"), dict)
        else {}
    )
    artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
    histogram = episode.get("action_histogram")
    action_total = _histogram_total(histogram)
    dominant_action, dominant_count = _dominant_histogram_item(histogram)
    dominant_action_share = dominant_count / action_total if action_total else None
    action_entropy = _normalized_histogram_entropy(histogram)
    nonzero_reward_steps = episode.get("nonzero_reward_steps")
    if isinstance(nonzero_reward_steps, list):
        nonzero_reward_count = len(nonzero_reward_steps)
        positive_reward_count = sum(
            1 for item in nonzero_reward_steps if float(item.get("reward", 0.0)) > 0.0
        )
    else:
        nonzero_reward_count = None
        positive_reward_count = None
    stock_manual_match = manual_vs_stock.get("actions_match_for_recorded_prefix")
    fallback_used = status.get("model_fallback_used")
    strict_load = status.get("strict_policy_model_load_ok")
    eval_cap_steps = episode.get("max_eval_steps")
    if eval_cap_steps is None:
        eval_cap_steps = config.get("max_eval_steps")
    steps_survived = status.get("steps_run")
    if steps_survived is None:
        steps_survived = episode.get("steps_run")
    survival_fraction = _survival_fraction(steps_survived, eval_cap_steps)
    return {
        "index": job.get("index"),
        "checkpoint": job.get("checkpoint_label"),
        "seed": job.get("seed"),
        "stock_steps_survived": stock_rollout.get("steps_run"),
        "steps_survived": steps_survived,
        "eval_cap_steps": eval_cap_steps,
        "survival_fraction": survival_fraction,
        "survived_to_cap": _survived_to_cap(steps_survived, eval_cap_steps),
        "ok": bool(result.get("ok")),
        "strict_load": strict_load,
        "stock_episode_length": stock_rollout.get("episode_length"),
        "stock_return": _stock_evaluator_return(stock_evaluator),
        "return": episode.get("total_reward"),
        "stock_nonzero_reward_count": stock_rollout.get("nonzero_reward_count"),
        "stock_positive_reward_count": stock_rollout.get("positive_reward_count"),
        "stock_negative_reward_count": stock_rollout.get("negative_reward_count"),
        "stock_done": stock_rollout.get("done"),
        "stock_truncated": stock_rollout.get("truncated"),
        "stock_terminal_reason": stock_rollout.get("terminal_reason"),
        "stock_action_histogram": stock_rollout.get("action_histogram"),
        "stock_ok": stock_evaluator.get("ok"),
        "nonzero_reward_count": nonzero_reward_count,
        "positive_reward_count": positive_reward_count,
        "action_histogram": histogram,
        "dominant_action": dominant_action,
        "dominant_action_share": (
            round(dominant_action_share, 6) if dominant_action_share is not None else None
        ),
        "action_entropy": action_entropy,
        "stock_manual_match": stock_manual_match,
        "stock_manual_status": manual_vs_stock.get("reason"),
        "fallback_used": fallback_used,
        "elapsed_sec": result.get("remote_elapsed_sec"),
        "episode_elapsed_sec": episode.get("elapsed_sec"),
        "artifact_ref": artifact.get("ref"),
        "checkpoint_ref": job.get("checkpoint_ref"),
        "verdict": _parallel_eval_verdict(
            ok=bool(result.get("ok")),
            fallback_used=fallback_used,
            stock_manual_match=stock_manual_match,
            dominant_action_share=dominant_action_share,
            positive_reward_count=positive_reward_count,
            total_reward=episode.get("total_reward"),
        ),
    }


def _single_summary(result: dict[str, Any]) -> dict[str, Any]:
    job = {
        "index": 0,
        "checkpoint_ref": result.get("config", {}).get("checkpoint_ref")
        if isinstance(result.get("config"), dict)
        else None,
        "checkpoint_label": (
            _checkpoint_label(result["config"]["checkpoint_ref"], index=0)
            if isinstance(result.get("config"), dict)
            and isinstance(result["config"].get("checkpoint_ref"), str)
            else None
        ),
    }
    table = [_parallel_table_row(job, result)]
    output_ref = (
        result.get("config", {}).get("output_ref")
        if isinstance(result.get("config"), dict)
        else None
    )
    return {
        "schema": "curvyzero_lightzero_visual_pong_eval_smoke_summary/v0",
        "ok": bool(result.get("ok")),
        "survival_aggregate_table": _survival_aggregate_table(table),
        "survival_table": _survival_table(table),
        "table": table,
        "output_refs": [output_ref] if isinstance(output_ref, str) else [],
    }


def _parallel_summary(
    *,
    manifest_result: dict[str, Any],
    manifest_ref: str | None,
    jobs: list[dict[str, Any]],
    table: list[dict[str, Any]],
    output_refs: list[str],
) -> dict[str, Any]:
    survival_table = _survival_table(table)
    survival_aggregate_table = _survival_aggregate_table(table)
    return {
        "schema": "curvyzero_lightzero_visual_pong_parallel_eval_summary/v0",
        "ok": bool(manifest_result.get("ok")),
        "survival_aggregate_table": survival_aggregate_table,
        "survival_table": survival_table,
        "table": table,
        "manifest": manifest_result,
        "manifest_ref": manifest_ref,
        "jobs": jobs,
        "output_refs": output_refs,
    }


def _table_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_to_plain(value), sort_keys=True, separators=(",", ":"))
    return str(value)


def _tsv_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_table_value(row.get(column)) for column in columns))
    return "\n".join(lines)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _checkpoint_sort_key(label: Any) -> tuple[int, int | str]:
    text = str(label)
    match = re.search(r"iteration_(\d+)", text)
    if match is not None:
        return (0, int(match.group(1)))
    return (1, text)


def _survival_aggregate_table(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row in table:
        checkpoint = str(row.get("checkpoint") or "")
        by_checkpoint.setdefault(checkpoint, []).append(row)

    aggregate_rows = []
    for checkpoint, rows in sorted(by_checkpoint.items(), key=lambda item: _checkpoint_sort_key(item[0])):
        stock_steps = [
            value
            for value in (_number(row.get("stock_steps_survived")) for row in rows)
            if value is not None
        ]
        manual_steps = [
            value
            for value in (_number(row.get("steps_survived")) for row in rows)
            if value is not None
        ]
        step_values = stock_steps or manual_steps
        stock_returns = [
            value
            for value in (_number(row.get("stock_return")) for row in rows)
            if value is not None
        ]
        manual_returns = [
            value
            for value in (_number(row.get("return")) for row in rows)
            if value is not None
        ]
        score_values = stock_returns or manual_returns
        aggregate_rows.append(
            {
                "checkpoint": checkpoint,
                "seeds": len(rows),
                "mean_steps": _mean(step_values),
                "median_steps": _median(step_values),
                "min_steps": min(step_values) if step_values else None,
                "max_steps": max(step_values) if step_values else None,
                "ok": sum(1 for row in rows if row.get("ok")),
                "mean_score": _mean(score_values),
            }
        )
    return aggregate_rows


def _survival_table(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint": row.get("checkpoint"),
            "seed": row.get("seed"),
            "stock_steps": row.get("stock_steps_survived"),
            "manual_steps": row.get("steps_survived"),
            "cap": row.get("eval_cap_steps"),
            "stock_return": row.get("stock_return"),
            "stock_terminal": row.get("stock_terminal_reason"),
            "ok": row.get("ok"),
            "strict": row.get("strict_load"),
            "artifact_ref": row.get("artifact_ref"),
        }
        for row in table
    ]


def _stock_evaluator_return(stock_evaluator: dict[str, Any]) -> float | None:
    eval_output = stock_evaluator.get("eval_output")
    if (
        isinstance(eval_output, (list, tuple))
        and len(eval_output) >= 2
        and isinstance(eval_output[1], dict)
    ):
        raw = eval_output[1].get("eval_episode_return_mean")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return None


def _histogram_total(histogram: Any) -> int:
    if not isinstance(histogram, dict):
        return 0
    total = 0
    for value in histogram.values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            pass
    return total


def _dominant_histogram_item(histogram: Any) -> tuple[str | None, int]:
    if not isinstance(histogram, dict) or not histogram:
        return None, 0
    best_key = None
    best_count = -1
    for key, value in histogram.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > best_count:
            best_key = str(key)
            best_count = count
    return best_key, max(best_count, 0)


def _normalized_histogram_entropy(histogram: Any) -> float | None:
    total = _histogram_total(histogram)
    if not isinstance(histogram, dict) or total <= 0:
        return None
    counts = []
    for value in histogram.values():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts.append(count)
    if len(counts) <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counts)
    return round(entropy / math.log(len(counts)), 6)


def _survival_fraction(steps_survived: Any, eval_cap_steps: Any) -> float | None:
    try:
        steps = float(steps_survived)
        cap = float(eval_cap_steps)
    except (TypeError, ValueError):
        return None
    if cap <= 0.0:
        return None
    return round(steps / cap, 6)


def _survival_rate(steps_survived: Any, eval_cap_steps: Any) -> float | None:
    return _survival_fraction(steps_survived, eval_cap_steps)


def _survived_to_cap(steps_survived: Any, eval_cap_steps: Any) -> bool | None:
    try:
        steps = float(steps_survived)
        cap = float(eval_cap_steps)
    except (TypeError, ValueError):
        return None
    if cap <= 0.0:
        return None
    return steps >= cap


def _parallel_eval_verdict(
    *,
    ok: bool,
    fallback_used: Any,
    stock_manual_match: Any,
    dominant_action_share: float | None,
    positive_reward_count: int | None,
    total_reward: Any,
) -> str:
    if not ok:
        return "eval_failed"
    if bool(fallback_used):
        return "invalid_fallback_used"
    if stock_manual_match is False:
        return "manual_stock_mismatch"
    if dominant_action_share is not None and dominant_action_share >= 0.98:
        return "collapsed_action"
    if positive_reward_count is not None and positive_reward_count > 0:
        return "has_positive_reward"
    try:
        if float(total_reward) < 0.0:
            return "negative_return"
    except (TypeError, ValueError):
        pass
    return "no_clear_signal"


def _tensor_stats(value: Any) -> dict[str, Any]:
    summary = _summarize_value(value)
    if hasattr(value, "detach"):
        import torch

        tensor = value.detach().cpu().float().reshape(-1)
        if int(tensor.numel()) > 0:
            summary.update(
                {
                    "min": float(torch.min(tensor).item()),
                    "max": float(torch.max(tensor).item()),
                    "mean": float(torch.mean(tensor).item()),
                }
            )
    return summary


def _float_or_plain(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return _to_plain(value)


def _summarize_observation(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return _summarize_value(observation)
    return {
        "keys": [str(key) for key in observation],
        "observation": _summarize_value(observation.get("observation")),
        "action_mask": _summarize_value(observation.get("action_mask")),
        "action_mask_values": _to_plain(observation.get("action_mask")),
        "to_play": _float_or_plain(observation.get("to_play")),
        "timestep": _float_or_plain(observation.get("timestep")),
    }


def _root_output(output: Any) -> Any:
    plain = _to_plain(output)
    if isinstance(plain, dict):
        for key in (0, "0"):
            if key in plain:
                return plain[key]
    return plain


def _compact_mcts_output(output: Any) -> dict[str, Any]:
    root = _root_output(output)
    if not isinstance(root, dict):
        return {"raw": root}
    keys = [
        "action",
        "visit_count_distribution",
        "visit_count_distributions",
        "visit_count_distribution_entropy",
        "visit_counts",
        "predicted_policy_logits",
        "policy_logits",
        "predicted_value",
        "searched_value",
        "value",
    ]
    compact = {key: root.get(key) for key in keys if key in root}
    compact["output_keys"] = sorted(str(key) for key in root.keys())
    return compact


def _discover_action_meanings(env: Any) -> dict[str, Any]:
    seen: set[int] = set()
    queue: list[tuple[str, Any]] = [("env", env)]
    attempts: list[dict[str, Any]] = []
    while queue:
        label, candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        candidate_type = type(candidate).__module__ + "." + type(candidate).__name__
        for method_name in ("get_action_meanings",):
            method = getattr(candidate, method_name, None)
            if callable(method):
                try:
                    meanings = [str(item) for item in method()]
                    return {
                        "ok": True,
                        "source": f"{label}.{method_name}()",
                        "object_type": candidate_type,
                        "meanings": meanings,
                    }
                except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
                    attempts.append(
                        {
                            "source": f"{label}.{method_name}()",
                            "object_type": candidate_type,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
        ale = getattr(candidate, "ale", None)
        method = getattr(ale, "getActionMeanings", None)
        if callable(method):
            try:
                meanings = [str(item) for item in method()]
                return {
                    "ok": True,
                    "source": f"{label}.ale.getActionMeanings()",
                    "object_type": candidate_type,
                    "meanings": meanings,
                }
            except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
                attempts.append(
                    {
                        "source": f"{label}.ale.getActionMeanings()",
                        "object_type": candidate_type,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        child_attrs = (
            "unwrapped",
            "env",
            "_env",
            "_gym_env",
            "_raw_env",
            "wrapped_env",
            "gym_env",
        )
        for attr in child_attrs:
            child = getattr(candidate, attr, None)
            if child is not None and id(child) not in seen:
                queue.append((f"{label}.{attr}", child))
    attrs = [
        name
        for name in dir(env)
        if any(fragment in name.lower() for fragment in ("env", "ale", "action"))
    ][:80]
    return {
        "ok": False,
        "attempts": attempts,
        "visited_objects": len(seen),
        "top_level_attr_sample": attrs,
    }


def _gym_action_meanings(env_id: str) -> dict[str, Any]:
    try:
        import gym

        env = gym.make(env_id)
        try:
            unwrapped = getattr(env, "unwrapped", env)
            method = getattr(unwrapped, "get_action_meanings", None)
            if not callable(method):
                ale = getattr(unwrapped, "ale", None)
                method = getattr(ale, "getActionMeanings", None)
            if not callable(method):
                return {"ok": False, "env_id": env_id, "error": "no action meaning method found"}
            return {
                "ok": True,
                "env_id": env_id,
                "source": "gym.make(...).unwrapped",
                "meanings": [str(item) for item in method()],
            }
        finally:
            if hasattr(env, "close"):
                env.close()
    except Exception as exc:  # pragma: no cover - remote runtime diagnosis.
        return {"ok": False, "env_id": env_id, **_exception_result(exc)}


def _required_observation_channels(policy: Any) -> int:
    cfg = getattr(policy, "_cfg", None)
    shape = getattr(getattr(cfg, "model", None), "observation_shape", None)
    if shape is None:
        return 4
    return int(shape[0])


def _initial_frame_stack(observation: dict[str, Any], *, required_channels: int) -> Any:
    import numpy as np

    frame = np.asarray(observation["observation"], dtype=np.float32)
    if frame.ndim == 2:
        frame = frame[None, :, :]
    if frame.shape[0] == required_channels:
        return frame
    if frame.shape[0] == 1:
        return np.repeat(frame, required_channels, axis=0)
    raise ValueError(
        f"cannot initialize {required_channels}-frame stack from observation shape {list(frame.shape)}"
    )


def _advance_frame_stack(previous_stack: Any, observation: dict[str, Any], *, required_channels: int) -> Any:
    import numpy as np

    frame = np.asarray(observation["observation"], dtype=np.float32)
    if frame.ndim == 2:
        frame = frame[None, :, :]
    if frame.shape[0] == required_channels:
        return frame
    if frame.shape[0] == 1:
        return np.concatenate([previous_stack[1:], frame], axis=0)
    if frame.shape[0] < required_channels:
        keep = required_channels - frame.shape[0]
        return np.concatenate([previous_stack[-keep:], frame], axis=0)
    return frame[-required_channels:]


def _policy_observation(raw_observation: dict[str, Any], frame_stack: Any) -> dict[str, Any]:
    observation = dict(raw_observation)
    observation["observation"] = frame_stack
    return observation


def _timestep_parts(timestep: Any) -> tuple[Any, Any, bool, Any]:
    if hasattr(timestep, "obs"):
        return timestep.obs, timestep.reward, bool(timestep.done), timestep.info
    if isinstance(timestep, (list, tuple)) and len(timestep) >= 4:
        obs, reward, done, info = timestep[:4]
        return obs, reward, bool(done), info
    raise TypeError(f"unsupported timestep type: {type(timestep).__name__}")


def _extract_eval_action(output: Any) -> int:
    if isinstance(output, dict):
        if "action" in output:
            return int(_float_or_plain(output["action"]))
        if 0 in output:
            return _extract_eval_action(output[0])
        if "0" in output:
            return _extract_eval_action(output["0"])
    if isinstance(output, (list, tuple)) and output:
        return _extract_eval_action(output[0])
    raise ValueError(f"could not extract action from policy eval output: {output!r}")


def _policy_model_device(policy: Any) -> Any:
    import torch

    model = getattr(policy, "_model", None)
    if model is None:
        return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _policy_eval_action(policy: Any, observation: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    obs_tensor = torch.as_tensor(
        np.asarray([observation["observation"]]),
        dtype=torch.float32,
        device=_policy_model_device(policy),
    )
    action_mask = np.asarray([observation["action_mask"]], dtype=np.float32)
    to_play = [int(np.asarray(observation.get("to_play", -1)).reshape(-1)[0])]
    ready_env_id = np.asarray([0])
    with torch.no_grad():
        output = policy.eval_mode.forward(
            obs_tensor,
            action_mask=action_mask,
            to_play=to_play,
            ready_env_id=ready_env_id,
        )
    return {
        "ok": True,
        "source": "policy_eval_mode",
        "action": _extract_eval_action(output),
        "call": {
            "api": "MuZeroPolicy.eval_mode.forward",
            "data_shape": [int(item) for item in obs_tensor.shape],
            "action_mask_shape": [int(item) for item in action_mask.shape],
            "to_play": to_play,
            "ready_env_id": [0],
        },
        "compact_output": _compact_mcts_output(output),
        "output": _to_plain(output),
    }


def _model_greedy_action(policy: Any, observation: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    model = getattr(policy, "_model", None)
    if model is None:
        raise AttributeError("MuZeroPolicy has no _model attribute for fallback")
    obs_tensor = torch.as_tensor(
        np.asarray([observation["observation"]]),
        dtype=torch.float32,
        device=_policy_model_device(policy),
    )
    mask = np.asarray(observation["action_mask"], dtype=np.float32)
    with torch.no_grad():
        network_output = model.initial_inference(obs_tensor)
    logits = network_output.policy_logits.detach().cpu().reshape(-1)
    masked_logits = logits.clone()
    for idx, allowed in enumerate(mask.tolist()):
        if not allowed:
            masked_logits[idx] = -float("inf")
    action = int(torch.argmax(masked_logits).item())
    return {
        "ok": True,
        "source": "policy_model_greedy_initial_inference",
        "action": action,
        "call": {
            "api": "MuZeroPolicy._model.initial_inference",
            "data_shape": [int(item) for item in obs_tensor.shape],
        },
        "policy_logits": [float(value) for value in logits.tolist()],
        "policy_logits_summary": _tensor_stats(logits),
    }


def _select_action(
    *,
    policy: Any,
    observation: dict[str, Any],
    allow_model_fallback: bool,
) -> dict[str, Any]:
    try:
        return _policy_eval_action(policy, observation)
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        policy_error = _exception_result(exc)
    if not allow_model_fallback:
        return {
            "ok": False,
            "source": "policy_eval_mode",
            "policy_eval_error": policy_error,
        }
    try:
        fallback = _model_greedy_action(policy, observation)
        fallback["policy_eval_error"] = policy_error
        return fallback
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        return {
            "ok": False,
            "source": "policy_eval_mode_then_model_greedy_fallback",
            "policy_eval_error": policy_error,
            "fallback_error": _exception_result(exc),
        }


def _make_policy_and_env(
    *,
    main_config: Any,
    create_config: Any,
    state_dict: dict[str, Any],
    seed: int,
    use_cuda: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    from ding.config import compile_config
    from ding.envs import get_vec_env_setting
    from lzero.policy.muzero import MuZeroPolicy

    cfg = compile_config(
        copy.deepcopy(main_config),
        seed=seed,
        auto=True,
        create_cfg=copy.deepcopy(create_config),
        save_cfg=False,
    )
    if not hasattr(cfg.policy, "device"):
        cfg.policy.device = "cuda" if use_cuda else "cpu"
    elif use_cuda and str(cfg.policy.device) == "cpu":
        cfg.policy.device = "cuda"
    elif not use_cuda:
        cfg.policy.device = "cpu"
    cfg.policy.cuda = use_cuda
    policy = MuZeroPolicy(cfg.policy)
    policy_model = getattr(policy, "_model", None)
    if policy_model is None:
        raise AttributeError("MuZeroPolicy has no _model attribute")
    load = _load_state_dict_probe(policy_model, state_dict)
    if hasattr(policy_model, "eval"):
        policy_model.eval()
    if not load["ok"] or not load.get("strict"):
        raise RuntimeError(f"strict policy model checkpoint load failed: {load}")

    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    one_env_cfg = evaluator_env_cfg[0] if evaluator_env_cfg else collector_env_cfg[0]
    try:
        env = env_fn(cfg=one_env_cfg)
    except TypeError:
        env = env_fn(one_env_cfg)
    if hasattr(env, "seed"):
        env.seed(seed, dynamic_seed=False)
    surface = {
        "compiled_policy": {
            "cuda": bool(cfg.policy.cuda),
            "device": str(getattr(cfg.policy, "device", "missing")),
            "num_simulations": int(cfg.policy.num_simulations),
            "batch_size": int(cfg.policy.batch_size),
            "model_type": str(cfg.policy.model.model_type),
            "observation_shape": _to_plain(cfg.policy.model.observation_shape),
            "action_space_size": int(cfg.policy.model.action_space_size),
        },
        "env_factory": {
            "env_fn": getattr(env_fn, "__module__", "") + "." + getattr(env_fn, "__name__", repr(env_fn)),
            "collector_env_cfg_count": len(collector_env_cfg),
            "evaluator_env_cfg_count": len(evaluator_env_cfg),
            "env_type": type(env).__module__ + "." + type(env).__name__,
            "action_meanings": {
                "env_chain": _discover_action_meanings(env),
                "gym_direct": _gym_action_meanings(str(cfg.env.env_id)),
            },
        },
        "load_state_dict": load,
    }
    return policy, env, surface


def _run_eval_episode(
    *,
    policy: Any,
    env: Any,
    seed: int,
    max_eval_steps: int,
    step_detail_limit: int | None,
    allow_model_fallback: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    observation = env.reset()
    required_channels = _required_observation_channels(policy)
    frame_stack = _initial_frame_stack(observation, required_channels=required_channels)
    reset_observation_summary = _summarize_observation(observation)
    initial_policy_observation_summary = _summarize_observation(
        _policy_observation(observation, frame_stack)
    )
    steps: list[dict[str, Any]] = []
    step_summaries: list[dict[str, Any]] = []
    total_reward = 0.0
    policy_eval_step_count = 0
    fallback_step_count = 0
    done = False
    for step_index in range(max_eval_steps):
        stacked_observation = _policy_observation(observation, frame_stack)
        action_result = _select_action(
            policy=policy,
            observation=stacked_observation,
            allow_model_fallback=allow_model_fallback,
        )
        if not action_result["ok"]:
            failed_step = {
                "step_index": step_index,
                "ok": False,
                "before_env_observation": _summarize_observation(observation),
                "before_policy_observation": _summarize_observation(stacked_observation),
                "action_result": action_result,
            }
            steps.append(failed_step)
            step_summaries.append({"step_index": step_index, "ok": False, "action_result": action_result})
            break

        action = int(action_result["action"])
        env_step_started = time.perf_counter()
        timestep = env.step(action)
        env_step_elapsed_sec = time.perf_counter() - env_step_started
        next_observation, reward, done, info = _timestep_parts(timestep)
        reward_float = float(_float_or_plain(reward))
        total_reward += reward_float
        next_frame_stack = _advance_frame_stack(
            frame_stack,
            next_observation,
            required_channels=required_channels,
        )
        next_stacked_observation = _policy_observation(next_observation, next_frame_stack)
        if action_result["source"] == "policy_eval_mode":
            policy_eval_step_count += 1
        else:
            fallback_step_count += 1
        summary_step = {
            "step_index": step_index,
            "ok": True,
            "action": action,
            "action_source": action_result["source"],
            "mcts_output": action_result.get("compact_output"),
            "reward": reward_float,
            "done": bool(done),
            "env_step_elapsed_sec": round(env_step_elapsed_sec, 6),
            "before_env_observation_shape": _summarize_value(observation.get("observation")).get("shape"),
            "before_policy_observation_shape": _summarize_value(stacked_observation.get("observation")).get("shape"),
            "after_env_observation_shape": _summarize_value(next_observation.get("observation")).get("shape"),
            "after_policy_observation_shape": _summarize_value(next_stacked_observation.get("observation")).get("shape"),
            "info": _to_plain(info),
        }
        step_summaries.append(summary_step)
        if step_detail_limit is None or len(steps) < step_detail_limit:
            steps.append(
                {
                    "step_index": step_index,
                    "ok": True,
                    "action": action,
                    "action_source": action_result["source"],
                    "reward": reward_float,
                    "done": bool(done),
                    "before_env_observation": _summarize_observation(observation),
                    "before_policy_observation": _summarize_observation(stacked_observation),
                    "after_env_observation": _summarize_observation(next_observation),
                    "after_policy_observation": _summarize_observation(next_stacked_observation),
                    "info": _to_plain(info),
                    "action_result": action_result,
                }
            )
        observation = next_observation
        frame_stack = next_frame_stack
        if done:
            break

    close_result: dict[str, Any] = {"called": False}
    if hasattr(env, "close"):
        try:
            env.close()
            close_result = {"called": True, "ok": True}
        except Exception as exc:  # pragma: no cover - remote cleanup diagnosis.
            close_result = {"called": True, **_exception_result(exc)}

    ok_steps = [step for step in step_summaries if step["ok"]]
    actions = [int(step["action"]) for step in ok_steps]
    rewards = [float(step["reward"]) for step in ok_steps]
    nonzero_reward_steps = [
        {"step_index": int(step["step_index"]), "reward": float(step["reward"])}
        for step in ok_steps
        if float(step["reward"]) != 0.0
    ]
    terminal_steps = [
        {"step_index": int(step["step_index"]), "done": bool(step["done"]), "info": step.get("info")}
        for step in ok_steps
        if bool(step["done"])
    ]
    elapsed_sec = time.perf_counter() - started
    return {
        "ok": bool(ok_steps and all(step["ok"] for step in step_summaries)),
        "seed": seed,
        "max_eval_steps": max_eval_steps,
        "step_detail_limit": step_detail_limit,
        "required_observation_channels": required_channels,
        "reset_observation": reset_observation_summary,
        "initial_policy_observation": initial_policy_observation_summary,
        "steps_run": len(ok_steps),
        "done": bool(done),
        "total_reward": total_reward,
        "policy_eval_step_count": policy_eval_step_count,
        "fallback_step_count": fallback_step_count,
        "policy_could_act_in_real_env": bool(policy_eval_step_count > 0),
        "actions": actions,
        "action_histogram": dict(sorted(Counter(actions).items())),
        "rewards": rewards,
        "reward_histogram": dict(sorted(Counter(rewards).items())),
        "nonzero_reward_steps": nonzero_reward_steps,
        "terminal_steps": terminal_steps,
        "step_summaries": step_summaries,
        "steps": steps,
        "steps_truncated": bool(step_detail_limit is not None and len(steps) < len(step_summaries)),
        "close": close_result,
        "elapsed_sec": round(elapsed_sec, 6),
        "steps_per_sec": round(len(ok_steps) / elapsed_sec, 6) if elapsed_sec > 0 else None,
    }


def _make_env_thunk(env_fn: Any, env_cfg: Any) -> Any:
    def _thunk() -> Any:
        cfg_copy = copy.deepcopy(env_cfg)
        try:
            return env_fn(cfg=cfg_copy)
        except TypeError:
            return env_fn(cfg_copy)

    return _thunk


def _action_int_or_none(action: Any) -> int | None:
    plain = _to_plain(action)
    if isinstance(plain, dict):
        for key in ("action", 0, "0"):
            if key in plain:
                return _action_int_or_none(plain[key])
        return None
    if isinstance(plain, (list, tuple)):
        if len(plain) == 1:
            return _action_int_or_none(plain[0])
        return None
    try:
        return int(plain)
    except (TypeError, ValueError):
        return None


def _truncation_from_info(info: Any) -> bool | None:
    plain = _to_plain(info)
    if not isinstance(plain, dict):
        return None
    for key in ("TimeLimit.truncated", "truncated", "timeout", "TimeLimit.truncation"):
        if key in plain:
            return bool(plain[key])
    return None


def _terminal_reason_from_info(info: Any, *, done: bool, truncated: bool | None) -> str | None:
    plain = _to_plain(info)
    if isinstance(plain, dict):
        for key in ("terminal_reason", "end_reason", "episode_end_reason", "reason"):
            value = plain.get(key)
            if value is not None:
                return str(value)
    if truncated is True:
        return "truncated"
    if done:
        return "done"
    return None


def _record_stock_step(
    records: list[dict[str, Any]],
    record: dict[str, Any],
    record_path: str | None,
) -> None:
    plain = _to_plain(record)
    records.append(plain)
    if record_path is None:
        return
    try:
        with Path(record_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(plain, sort_keys=True) + "\n")
    except Exception:
        pass


def _load_stock_step_records(
    records: list[dict[str, Any]],
    record_path: str | None,
) -> list[dict[str, Any]]:
    if record_path is not None:
        path = Path(record_path)
        if path.exists():
            loaded = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        loaded.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if loaded:
                return loaded
    return records


def _make_recording_env_thunk(
    env_fn: Any,
    env_cfg: Any,
    records: list[dict[str, Any]],
    *,
    record_path: str | None,
) -> Any:
    def _thunk() -> Any:
        inner = _make_env_thunk(env_fn, env_cfg)()

        class _RecordingEnv:
            def __init__(self, wrapped: Any) -> None:
                self._wrapped = wrapped
                self._reset_count = 0

            def __getattr__(self, name: str) -> Any:
                return getattr(self._wrapped, name)

            def reset(self, *args: Any, **kwargs: Any) -> Any:
                self._reset_count += 1
                return self._wrapped.reset(*args, **kwargs)

            def step(self, action: Any) -> Any:
                step_index = len(records)
                started = time.perf_counter()
                timestep = self._wrapped.step(action)
                elapsed_sec = time.perf_counter() - started
                try:
                    _obs, reward, done, info = _timestep_parts(timestep)
                    reward_float = float(_float_or_plain(reward))
                    info_plain = _to_plain(info)
                    truncated = _truncation_from_info(info_plain)
                    _record_stock_step(
                        records,
                        {
                            "step_index": step_index,
                            "reset_count": self._reset_count,
                            "action": _action_int_or_none(action),
                            "action_raw": _to_plain(action),
                            "reward": reward_float,
                            "done": bool(done),
                            "truncated": truncated,
                            "terminal_reason": _terminal_reason_from_info(
                                info_plain,
                                done=bool(done),
                                truncated=truncated,
                            ),
                            "info": info_plain,
                            "env_step_elapsed_sec": round(elapsed_sec, 6),
                        },
                        record_path,
                    )
                except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
                    _record_stock_step(
                        records,
                        {
                            "step_index": step_index,
                            "action": _action_int_or_none(action),
                            "action_raw": _to_plain(action),
                            "recording_error": _exception_result(exc),
                            "env_step_elapsed_sec": round(elapsed_sec, 6),
                        },
                        record_path,
                    )
                return timestep

        return _RecordingEnv(inner)

    return _thunk


def _stock_rollout_summary(
    records: list[dict[str, Any]],
    *,
    max_eval_steps: int,
) -> dict[str, Any]:
    ok_steps = [step for step in records if "recording_error" not in step]
    actions = [
        int(step["action"])
        for step in ok_steps
        if step.get("action") is not None
    ]
    rewards = []
    for step in ok_steps:
        try:
            rewards.append(float(step.get("reward", 0.0)))
        except (TypeError, ValueError):
            pass
    terminal_steps = [step for step in ok_steps if bool(step.get("done"))]
    terminal_step = terminal_steps[-1] if terminal_steps else {}
    truncated_values = [
        bool(step["truncated"])
        for step in ok_steps
        if step.get("truncated") is not None
    ]
    done = bool(terminal_steps)
    truncated = bool(any(truncated_values)) if truncated_values else None
    terminal_reason = terminal_step.get("terminal_reason")
    return {
        "ok": bool(ok_steps) and len(ok_steps) == len(records),
        "source": "stock_env_wrapper_under_muzero_evaluator",
        "max_eval_steps": max_eval_steps,
        "steps_run": len(ok_steps),
        "episode_length": len(ok_steps),
        "done": done,
        "truncated": truncated,
        "terminal_reason": terminal_reason,
        "terminal_info": terminal_step.get("info"),
        "total_reward": round(sum(rewards), 6) if rewards else 0.0,
        "nonzero_reward_count": sum(1 for reward in rewards if reward != 0.0),
        "positive_reward_count": sum(1 for reward in rewards if reward > 0.0),
        "negative_reward_count": sum(1 for reward in rewards if reward < 0.0),
        "nonzero_reward_steps": [
            {"step_index": int(step["step_index"]), "reward": float(step["reward"])}
            for step in ok_steps
            if float(step.get("reward", 0.0)) != 0.0
        ],
        "actions": actions,
        "action_histogram": dict(sorted(Counter(actions).items())),
        "reward_histogram": dict(sorted(Counter(rewards).items())),
        "step_records": records,
        "recording_error_count": len(records) - len(ok_steps),
    }


def _run_stock_evaluator_probe(
    *,
    main_config: Any,
    create_config: Any,
    state_dict: dict[str, Any],
    seed: int,
    max_eval_steps: int,
    use_cuda: bool,
    optimizer_phase_timing: bool = DEFAULT_OPTIMIZER_PHASE_TIMING,
) -> dict[str, Any]:
    started = time.perf_counter()
    phase_timing: dict[str, float] | None = {} if optimizer_phase_timing else None
    records: list[dict[str, Any]] = []
    stock_step_records: list[dict[str, Any]] = []
    stock_step_record_path: Path | None = None
    env_manager_patch: dict[str, Any] | None = None
    env_manager = None
    try:
        with _optimizer_timed_phase(phase_timing, "stock_config_model_policy_setup_sec"):
            import inspect
            import tempfile

            from ding.config import compile_config
            from ding.envs import create_env_manager, get_vec_env_setting
            from lzero.worker import MuZeroEvaluator
            from lzero.policy.muzero import MuZeroPolicy

            cfg = compile_config(
                copy.deepcopy(main_config),
                seed=seed,
                auto=True,
                create_cfg=copy.deepcopy(create_config),
                save_cfg=False,
            )
            if not hasattr(cfg.policy, "device"):
                cfg.policy.device = "cuda" if use_cuda else "cpu"
            elif use_cuda and str(cfg.policy.device) == "cpu":
                cfg.policy.device = "cuda"
            elif not use_cuda:
                cfg.policy.device = "cpu"
            cfg.policy.cuda = use_cuda
            if hasattr(cfg.env, "evaluator_env_num"):
                cfg.env.evaluator_env_num = 1
            if hasattr(cfg.env, "n_evaluator_episode"):
                cfg.env.n_evaluator_episode = 1
            if hasattr(cfg.env, "eval_max_episode_steps"):
                cfg.env.eval_max_episode_steps = max_eval_steps
            manager_cfg = getattr(cfg.env, "manager", None)
            if isinstance(manager_cfg, dict):
                old_manager_type = manager_cfg.get("type")
                manager_cfg["type"] = "base"
                env_manager_patch = {
                    "path": "env.manager.type",
                    "old": _to_plain(old_manager_type),
                    "new": "base",
                    "reason": (
                        "Eval uses one in-process env to avoid Atari subprocess "
                        "banner output corrupting multiprocessing pipes."
                    ),
                }
            elif manager_cfg is not None and hasattr(manager_cfg, "type"):
                old_manager_type = manager_cfg.type
                manager_cfg.type = "base"
                env_manager_patch = {
                    "path": "env.manager.type",
                    "old": _to_plain(old_manager_type),
                    "new": "base",
                    "reason": (
                        "Eval uses one in-process env to avoid Atari subprocess "
                        "banner output corrupting multiprocessing pipes."
                    ),
                }

            policy = MuZeroPolicy(cfg.policy)
            policy_model = getattr(policy, "_model", None)
            if policy_model is None:
                raise AttributeError("MuZeroPolicy has no _model attribute")
            load = _load_state_dict_probe(policy_model, state_dict)
            if hasattr(policy_model, "eval"):
                policy_model.eval()
            if not load["ok"] or not load.get("strict"):
                raise RuntimeError(f"strict policy model checkpoint load failed: {load}")

        class _RecordingEvalMode:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                output = self._inner.forward(*args, **kwargs)
                if len(records) < 32:
                    data = args[0] if args else kwargs.get("data")
                    action_mask = (
                        args[1]
                        if len(args) > 1
                        else kwargs.get("action_mask")
                    )
                    to_play = args[2] if len(args) > 2 else kwargs.get("to_play")
                    records.append(
                        {
                            "call_index": len(records),
                            "data": _summarize_value(data),
                            "action_mask": _summarize_value(action_mask),
                            "action_mask_values": _to_plain(action_mask),
                            "ready_env_id": _to_plain(kwargs.get("ready_env_id")),
                            "to_play": _to_plain(to_play),
                            "timestep": _to_plain(kwargs.get("timestep")),
                            "action": _extract_eval_action(output),
                            "mcts_output": _compact_mcts_output(output),
                        }
                    )
                return output

        with _optimizer_timed_phase(phase_timing, "stock_env_evaluator_setup_sec"):
            recording_eval_mode = _RecordingEvalMode(policy.eval_mode)
            env_fn, _collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
            selected_env_cfg = evaluator_env_cfg[:1]
            if not selected_env_cfg:
                raise RuntimeError("get_vec_env_setting returned no evaluator env configs")
            temp_root = Path(tempfile.mkdtemp(prefix="curvyzero_stock_eval_"))
            stock_step_record_path = temp_root / "stock_step_records.jsonl"
            env_fns = [
                _make_recording_env_thunk(
                    env_fn,
                    env_cfg,
                    stock_step_records,
                    record_path=str(stock_step_record_path),
                )
                for env_cfg in selected_env_cfg
            ]
            env_manager = create_env_manager(cfg.env.manager, env_fns)
            if hasattr(env_manager, "seed"):
                env_manager.seed(seed, dynamic_seed=False)

            constructor_signature = str(inspect.signature(MuZeroEvaluator))
            evaluator = MuZeroEvaluator(
                eval_freq=int(cfg.policy.eval_freq),
                n_evaluator_episode=1,
                stop_value=float(getattr(cfg.env, "stop_value", 1e6)),
                env=env_manager,
                policy=recording_eval_mode,
                tb_logger=None,
                exp_name=str(temp_root),
                instance_name="curvyzero_stock_eval_parity",
                policy_config=cfg.policy,
            )
        eval_attempts: list[dict[str, Any]] = []

        def save_ckpt_fn(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        eval_output = None
        eval_ok = False
        with _optimizer_timed_phase(phase_timing, "stock_evaluator_eval_sec"):
            for label, call in (
                ("keyword_train_iter_envstep", lambda: evaluator.eval(save_ckpt_fn, train_iter=0, envstep=0)),
                ("positional_save_train_envstep", lambda: evaluator.eval(save_ckpt_fn, 0, 0)),
                ("positional_train_envstep_save", lambda: evaluator.eval(0, 0, save_ckpt_fn)),
            ):
                try:
                    eval_output = call()
                    eval_attempts.append({"label": label, "ok": True})
                    eval_ok = True
                    break
                except Exception as exc:  # pragma: no cover - remote API diagnosis.
                    eval_attempts.append(
                        {
                            "label": label,
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            if not eval_ok:
                raise RuntimeError(f"stock evaluator eval calls failed: {eval_attempts}")
            stock_step_records = _load_stock_step_records(
                stock_step_records,
                str(stock_step_record_path),
            )
        result = {
            "ok": True,
            "path": STOCK_EVALUATOR_PATH,
            "generic_evaluator_path_not_used": GENERIC_EVALUATOR_PATH,
            "generic_evaluator_reason": (
                "LightZero train_muzero wires lzero.worker.MuZeroEvaluator for MuZero; "
                "the generic DI-engine interaction evaluator only forwards observation data "
                "and does not provide MuZeroPolicy._forward_eval's required action_mask."
            ),
            "constructor_signature": constructor_signature,
            "eval_attempts": eval_attempts,
            "eval_output": _to_plain(eval_output),
            "stock_rollout": _stock_rollout_summary(
                stock_step_records,
                max_eval_steps=max_eval_steps,
            ),
            "env_manager_patch": env_manager_patch,
            "recorded_eval_mode_calls": records,
            "recorded_call_count": len(records),
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
        if phase_timing is not None:
            result["optimizer_phase_timing_sec"] = phase_timing
        return result
    except Exception as exc:  # pragma: no cover - remote API diagnosis.
        stock_step_records = _load_stock_step_records(
            stock_step_records,
            str(stock_step_record_path) if stock_step_record_path is not None else None,
        )
        result = {
            "ok": False,
            "path": STOCK_EVALUATOR_PATH,
            "generic_evaluator_path_not_used": GENERIC_EVALUATOR_PATH,
            "blocker": _exception_result(exc),
            "stock_rollout": _stock_rollout_summary(
                stock_step_records,
                max_eval_steps=max_eval_steps,
            ),
            "env_manager_patch": env_manager_patch,
            "recorded_eval_mode_calls": records,
            "recorded_call_count": len(records),
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
        if phase_timing is not None:
            result["optimizer_phase_timing_sec"] = phase_timing
        return result
    finally:
        if env_manager is not None and hasattr(env_manager, "close"):
            try:
                env_manager.close()
            except Exception:
                pass


def _compare_manual_and_stock(manual_episode: dict[str, Any], stock_probe: dict[str, Any] | None) -> dict[str, Any]:
    if manual_episode.get("skipped"):
        return {
            "ok": True,
            "reason": "manual rollout skipped; stock evaluator is the primary readout",
            "stock_env_steps_run": (
                stock_probe.get("stock_rollout", {}).get("steps_run")
                if isinstance(stock_probe, dict)
                and isinstance(stock_probe.get("stock_rollout"), dict)
                else None
            ),
        }
    if not stock_probe:
        return {"ok": False, "reason": "stock evaluator probe not requested"}
    if not stock_probe.get("ok"):
        return {
            "ok": False,
            "reason": "stock evaluator probe did not complete",
            "blocker": stock_probe.get("blocker"),
        }
    manual_actions = [int(item) for item in manual_episode.get("actions", [])[:32]]
    stock_actions = [
        int(item["action"])
        for item in stock_probe.get("recorded_eval_mode_calls", [])
        if "action" in item
    ][:32]
    stock_rollout = (
        stock_probe.get("stock_rollout")
        if isinstance(stock_probe.get("stock_rollout"), dict)
        else {}
    )
    return {
        "ok": bool(stock_actions),
        "manual_first32_actions": manual_actions,
        "stock_first32_actions": stock_actions,
        "actions_match_for_recorded_prefix": manual_actions[: len(stock_actions)] == stock_actions,
        "manual_action_histogram": manual_episode.get("action_histogram"),
        "stock_action_histogram": dict(sorted(Counter(stock_actions).items())),
        "stock_env_action_histogram": stock_rollout.get("action_histogram"),
        "stock_env_steps_run": stock_rollout.get("steps_run"),
        "stock_env_positive_reward_count": stock_rollout.get("positive_reward_count"),
        "stock_env_negative_reward_count": stock_rollout.get("negative_reward_count"),
    }


def _eval_checkpoint(
    *,
    checkpoint_path: Path,
    run_id: str,
    attempt_id: str,
    env_id: str,
    seed: int,
    max_env_step: int,
    max_train_iter: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    max_episode_steps: int,
    game_segment_length: int,
    max_eval_steps: int,
    step_detail_limit: int | None,
    allow_model_fallback: bool,
    run_stock_evaluator: bool,
    skip_manual_rollout: bool,
    use_cuda: bool,
    optimizer_phase_timing: bool = DEFAULT_OPTIMIZER_PHASE_TIMING,
) -> dict[str, Any]:
    if skip_manual_rollout and not run_stock_evaluator:
        raise ValueError("skip_manual_rollout requires run_stock_evaluator")
    phase_timing: dict[str, float] | None = {} if optimizer_phase_timing else None
    with _optimizer_timed_phase(phase_timing, "checkpoint_load_sec"):
        checkpoint = _torch_load(checkpoint_path)
        state_candidate = _find_state_dict(checkpoint)
        if state_candidate is None:
            raise ValueError("no tensor state dict found under common LightZero checkpoint keys")
        state_path, state_dict = state_candidate
    with _optimizer_timed_phase(phase_timing, "config_setup_sec"):
        patched = _patched_stock_atari_pong_configs(
            env_id=env_id,
            seed=seed,
            run_id=run_id,
            attempt_id=attempt_id,
            max_env_step=max_env_step,
            max_train_iter=max_train_iter,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            num_simulations=num_simulations,
            batch_size=batch_size,
            update_per_collect=update_per_collect,
            max_episode_steps=max_episode_steps,
            game_segment_length=game_segment_length,
            use_cuda=use_cuda,
        )
    if skip_manual_rollout:
        surface = {
            "manual_rollout": {
                "skipped": True,
                "reason": (
                    "stock-only triage skips the duplicate manual policy/env rollout; "
                    "the stock MuZeroEvaluator below still strict-loads the checkpoint "
                    "and runs the LightZero evaluator env."
                ),
            }
        }
        episode = {
            "ok": True,
            "skipped": True,
            "seed": seed,
            "max_eval_steps": max_eval_steps,
            "step_detail_limit": step_detail_limit,
            "steps_run": None,
            "done": None,
            "total_reward": None,
            "policy_eval_step_count": 0,
            "fallback_step_count": 0,
            "policy_could_act_in_real_env": None,
            "actions": [],
            "action_histogram": {},
            "rewards": [],
            "reward_histogram": {},
            "nonzero_reward_steps": [],
            "terminal_steps": [],
            "step_summaries": [],
            "steps": [],
            "steps_truncated": False,
            "elapsed_sec": 0.0,
            "steps_per_sec": None,
        }
    else:
        with _optimizer_timed_phase(phase_timing, "manual_policy_env_setup_sec"):
            policy, env, surface = _make_policy_and_env(
                main_config=patched["main_config"],
                create_config=patched["create_config"],
                state_dict=state_dict,
                seed=seed,
                use_cuda=use_cuda,
            )
        with _optimizer_timed_phase(phase_timing, "manual_eval_episode_sec"):
            episode = _run_eval_episode(
                policy=policy,
                env=env,
                seed=seed,
                max_eval_steps=max_eval_steps,
                step_detail_limit=step_detail_limit,
                allow_model_fallback=allow_model_fallback,
            )
    stock_evaluator = None
    if run_stock_evaluator:
        with _optimizer_timed_phase(phase_timing, "stock_evaluator_total_sec"):
            stock_evaluator = _run_stock_evaluator_probe(
                main_config=patched["main_config"],
                create_config=patched["create_config"],
                state_dict=state_dict,
                seed=seed,
                max_eval_steps=max_eval_steps,
                use_cuda=use_cuda,
                optimizer_phase_timing=optimizer_phase_timing,
            )
    result = {
        "schema": "curvyzero_lightzero_visual_pong_eval_smoke/v0",
        "checkpoint": _checkpoint_summary(checkpoint_path, include_sha256=False),
        "load": {
            "ok": True,
            "state_dict_path": state_path,
            "tensor_count": len(state_dict),
            "keys_sample": list(state_dict)[:40],
        },
        "stock_example": {
            "task": env_id,
            "algorithm": "MuZero",
            "module": patched["module"],
            "trainer_entrypoint": "lzero.entry.train_muzero",
            "original_surface": patched["original_surface"],
            "patched_surface": patched["patched_surface"],
            "patches": patched["patches"],
        },
        "surface": surface,
        "episode": episode,
        "stock_evaluator": stock_evaluator,
        "manual_vs_stock": _compare_manual_and_stock(episode, stock_evaluator),
        "eval_mode": "stock_only" if skip_manual_rollout else "manual_plus_stock",
    }
    if phase_timing is not None:
        if isinstance(stock_evaluator, dict):
            stock_timing = stock_evaluator.get("optimizer_phase_timing_sec")
            if isinstance(stock_timing, dict):
                for key, value in stock_timing.items():
                    phase_timing[key] = value
        result["optimizer_phase_timing_sec"] = phase_timing
    return result


def _run_lightzero_pong_eval_smoke(
    *,
    compute: str,
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    allow_model_fallback: bool = DEFAULT_ALLOW_MODEL_FALLBACK,
    run_stock_evaluator: bool = DEFAULT_RUN_STOCK_EVALUATOR,
    skip_manual_rollout: bool = DEFAULT_SKIP_MANUAL_ROLLOUT,
    emit_result_json: bool = True,
    quiet_framework_logs: bool = DEFAULT_QUIET_FRAMEWORK_LOGS,
    optimizer_phase_timing: bool = DEFAULT_OPTIMIZER_PHASE_TIMING,
) -> dict[str, Any]:
    if skip_manual_rollout and not run_stock_evaluator:
        raise ValueError("skip_manual_rollout requires run_stock_evaluator")
    if compute not in {
        "cpu",
        "gpu-l4-t4",
        GPU_L4_T4_CPU8_COMPUTE,
        GPU_L4_T4_CPU40_COMPUTE,
    }:
        raise ValueError(
            f"unknown compute: {compute!r}; expected 'cpu', 'gpu-l4-t4', "
            f"{GPU_L4_T4_CPU8_COMPUTE!r}, or {GPU_L4_T4_CPU40_COMPUTE!r}"
        )
    use_cuda = compute in {
        "gpu-l4-t4",
        GPU_L4_T4_CPU8_COMPUTE,
        GPU_L4_T4_CPU40_COMPUTE,
    }
    started = time.perf_counter()
    started_at = runs.utc_timestamp()
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    output_ref = output_ref or _default_output_ref(run_id=run_id, attempt_id=attempt_id)
    output_path = runs.volume_path(RUNS_MOUNT, output_ref)
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "gym": _version_or_missing("gym"),
        "gymnasium": _version_or_missing("gymnasium"),
        "ale-py": _version_or_missing("ale-py", "ale_py"),
        "opencv-python-headless": _version_or_missing("opencv-python-headless"),
        "AutoROM": _version_or_missing("AutoROM"),
    }
    config = {
        "job_kind": "lightzero_official_visual_pong_eval_smoke",
        "compute": compute,
        "use_cuda": use_cuda,
        "runtime_compute": _runtime_compute_summary(requested_compute=compute),
        "checkpoint_ref": checkpoint_ref,
        "output_ref": output_ref,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "env_id": env_id,
        "seed": seed,
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "update_per_collect": update_per_collect,
        "max_episode_steps": max_episode_steps,
        "game_segment_length": game_segment_length,
        "max_eval_steps": max_eval_steps,
        "step_detail_limit": step_detail_limit,
        "allow_model_fallback": allow_model_fallback,
        "run_stock_evaluator": run_stock_evaluator,
        "skip_manual_rollout": skip_manual_rollout,
        "optimizer_phase_timing": optimizer_phase_timing,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    cap_warning = _eval_cap_warning(
        max_episode_steps=max_episode_steps,
        max_eval_steps=max_eval_steps,
    )
    if cap_warning is not None:
        config["warnings"] = [cap_warning]
    try:
        with _quiet_framework_output(quiet_framework_logs):
            eval_result = _eval_checkpoint(
                checkpoint_path=checkpoint_path,
                run_id=run_id,
                attempt_id=attempt_id,
                env_id=env_id,
                seed=seed,
                max_env_step=max_env_step,
                max_train_iter=max_train_iter,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                num_simulations=num_simulations,
                batch_size=batch_size,
                update_per_collect=update_per_collect,
                max_episode_steps=max_episode_steps,
                game_segment_length=game_segment_length,
                max_eval_steps=max_eval_steps,
                step_detail_limit=step_detail_limit,
                allow_model_fallback=allow_model_fallback,
                run_stock_evaluator=run_stock_evaluator,
                skip_manual_rollout=skip_manual_rollout,
                use_cuda=use_cuda,
                optimizer_phase_timing=optimizer_phase_timing,
            )
        episode = eval_result["episode"]
        stock_evaluator = (
            eval_result.get("stock_evaluator")
            if isinstance(eval_result.get("stock_evaluator"), dict)
            else {}
        )
        stock_rollout = (
            stock_evaluator.get("stock_rollout")
            if isinstance(stock_evaluator.get("stock_rollout"), dict)
            else {}
        )
        stock_ok = bool(stock_evaluator.get("ok"))
        manual_ok = bool(episode["ok"] and episode["policy_could_act_in_real_env"])
        result_ok = stock_ok if skip_manual_rollout else manual_ok
        status_steps_run = (
            stock_rollout.get("steps_run") if skip_manual_rollout else int(episode["steps_run"])
        )
        result: dict[str, Any] = {
            **eval_result,
            "ok": result_ok,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "status": {
                "checkpoint_load_ok": True,
                "strict_policy_model_load_ok": True,
                "env_reset_ok": bool(stock_rollout.get("steps_run")) if skip_manual_rollout else bool(episode["steps"]),
                "policy_could_act_in_real_env": (
                    bool(stock_evaluator.get("recorded_call_count"))
                    if skip_manual_rollout
                    else bool(episode["policy_could_act_in_real_env"])
                ),
                "model_fallback_used": bool(episode["fallback_step_count"] > 0),
                "steps_run": status_steps_run,
            },
        }
    except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
        result = {
            "schema": "curvyzero_lightzero_visual_pong_eval_smoke/v0",
            "ok": False,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "checkpoint": _checkpoint_summary(checkpoint_path, include_sha256=False),
            "status": {
                "checkpoint_load_ok": False,
                "strict_policy_model_load_ok": False,
                "env_reset_ok": False,
                "policy_could_act_in_real_env": False,
                "model_fallback_used": False,
                "steps_run": 0,
            },
            "wrapper_error": _exception_result(exc),
        }

    phase_timing = (
        result.setdefault("optimizer_phase_timing_sec", {})
        if optimizer_phase_timing
        else None
    )
    with _optimizer_timed_phase(phase_timing, "artifact_write_sec"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        runs.write_json(output_path, _to_plain(result))
        result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    with _optimizer_timed_phase(phase_timing, "volume_commit_sec"):
        runs_volume.commit()
    if emit_result_json:
        print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=EVAL_FUNCTION_TIMEOUT_SEC,
    cpu=1.0,
)
def lightzero_pong_eval_smoke(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    allow_model_fallback: bool = DEFAULT_ALLOW_MODEL_FALLBACK,
    run_stock_evaluator: bool = DEFAULT_RUN_STOCK_EVALUATOR,
    skip_manual_rollout: bool = DEFAULT_SKIP_MANUAL_ROLLOUT,
    emit_result_json: bool = True,
    quiet_framework_logs: bool = DEFAULT_QUIET_FRAMEWORK_LOGS,
    optimizer_phase_timing: bool = DEFAULT_OPTIMIZER_PHASE_TIMING,
) -> dict[str, Any]:
    return _run_lightzero_pong_eval_smoke(
        compute="cpu",
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        allow_model_fallback=allow_model_fallback,
        run_stock_evaluator=run_stock_evaluator,
        skip_manual_rollout=skip_manual_rollout,
        emit_result_json=emit_result_json,
        quiet_framework_logs=quiet_framework_logs,
        optimizer_phase_timing=optimizer_phase_timing,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=EVAL_FUNCTION_TIMEOUT_SEC,
    cpu=2.0,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_pong_eval_smoke_gpu(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    allow_model_fallback: bool = DEFAULT_ALLOW_MODEL_FALLBACK,
    run_stock_evaluator: bool = DEFAULT_RUN_STOCK_EVALUATOR,
    skip_manual_rollout: bool = DEFAULT_SKIP_MANUAL_ROLLOUT,
    emit_result_json: bool = True,
    quiet_framework_logs: bool = DEFAULT_QUIET_FRAMEWORK_LOGS,
    optimizer_phase_timing: bool = DEFAULT_OPTIMIZER_PHASE_TIMING,
) -> dict[str, Any]:
    return _run_lightzero_pong_eval_smoke(
        compute="gpu-l4-t4",
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        allow_model_fallback=allow_model_fallback,
        run_stock_evaluator=run_stock_evaluator,
        skip_manual_rollout=skip_manual_rollout,
        emit_result_json=emit_result_json,
        quiet_framework_logs=quiet_framework_logs,
        optimizer_phase_timing=optimizer_phase_timing,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=EVAL_FUNCTION_TIMEOUT_SEC,
    cpu=8.0,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_pong_eval_smoke_gpu_cpu8(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    allow_model_fallback: bool = DEFAULT_ALLOW_MODEL_FALLBACK,
    run_stock_evaluator: bool = DEFAULT_RUN_STOCK_EVALUATOR,
    skip_manual_rollout: bool = DEFAULT_SKIP_MANUAL_ROLLOUT,
    emit_result_json: bool = True,
    quiet_framework_logs: bool = DEFAULT_QUIET_FRAMEWORK_LOGS,
    optimizer_phase_timing: bool = DEFAULT_OPTIMIZER_PHASE_TIMING,
) -> dict[str, Any]:
    return _run_lightzero_pong_eval_smoke(
        compute=GPU_L4_T4_CPU8_COMPUTE,
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        allow_model_fallback=allow_model_fallback,
        run_stock_evaluator=run_stock_evaluator,
        skip_manual_rollout=skip_manual_rollout,
        emit_result_json=emit_result_json,
        quiet_framework_logs=quiet_framework_logs,
        optimizer_phase_timing=optimizer_phase_timing,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=EVAL_FUNCTION_TIMEOUT_SEC,
    cpu=40.0,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_pong_eval_smoke_gpu_cpu40(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    allow_model_fallback: bool = DEFAULT_ALLOW_MODEL_FALLBACK,
    run_stock_evaluator: bool = DEFAULT_RUN_STOCK_EVALUATOR,
    skip_manual_rollout: bool = DEFAULT_SKIP_MANUAL_ROLLOUT,
    emit_result_json: bool = True,
    quiet_framework_logs: bool = DEFAULT_QUIET_FRAMEWORK_LOGS,
    optimizer_phase_timing: bool = DEFAULT_OPTIMIZER_PHASE_TIMING,
) -> dict[str, Any]:
    return _run_lightzero_pong_eval_smoke(
        compute=GPU_L4_T4_CPU40_COMPUTE,
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        allow_model_fallback=allow_model_fallback,
        run_stock_evaluator=run_stock_evaluator,
        skip_manual_rollout=skip_manual_rollout,
        emit_result_json=emit_result_json,
        quiet_framework_logs=quiet_framework_logs,
        optimizer_phase_timing=optimizer_phase_timing,
    )


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=2 * 60, cpu=0.25)
def lightzero_pong_eval_parallel_manifest(
    manifest_ref: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    output_path = runs.volume_path(RUNS_MOUNT, manifest_ref)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(output_path, _to_plain(manifest))
    summary = runs.file_summary(output_path, mount=RUNS_MOUNT)
    runs_volume.commit()
    return {"ok": True, "manifest_ref": manifest_ref, "artifact": summary}


@app.local_entrypoint()
def main(
    compute: str = DEFAULT_COMPUTE,
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    checkpoint_refs: str | None = None,
    checkpoint_ref_template: str | None = None,
    selected_iterations: str | None = None,
    output_ref: str | None = None,
    manifest_ref: str | None = None,
    parallel: bool = False,
    eval_id: str = DEFAULT_PARALLEL_EVAL_ID,
    eval_pass: str = "custom",
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    eval_seeds: str | None = None,
    eval_seed_count: int | None = None,
    eval_seed_rng_seed: int = DEFAULT_EVAL_SEED_RNG_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    low_detail_max_eval_steps: int = DEFAULT_LOW_DETAIL_MAX_EVAL_STEPS,
    low_detail_step_detail_limit: int = DEFAULT_LOW_DETAIL_STEP_DETAIL_LIMIT,
    high_detail_max_eval_steps: int = DEFAULT_HIGH_DETAIL_MAX_EVAL_STEPS,
    high_detail_step_detail_limit: int = DEFAULT_HIGH_DETAIL_STEP_DETAIL_LIMIT,
    allow_model_fallback: bool = DEFAULT_ALLOW_MODEL_FALLBACK,
    run_stock_evaluator: bool = DEFAULT_RUN_STOCK_EVALUATOR,
    stock_only: bool = False,
    skip_manual_rollout: bool = DEFAULT_SKIP_MANUAL_ROLLOUT,
    summary_only: bool = DEFAULT_SUMMARY_ONLY,
    slim_manifest: bool = False,
    quiet_framework_logs: bool = DEFAULT_QUIET_FRAMEWORK_LOGS,
    optimizer_phase_timing: bool = DEFAULT_OPTIMIZER_PHASE_TIMING,
) -> None:
    effective_skip_manual_rollout = bool(skip_manual_rollout or stock_only)
    if effective_skip_manual_rollout:
        run_stock_evaluator = True
    if effective_skip_manual_rollout and not run_stock_evaluator:
        raise ValueError("skip_manual_rollout requires run_stock_evaluator")
    if compute == "cpu":
        eval_fn = lightzero_pong_eval_smoke
    elif compute == "gpu-l4-t4":
        eval_fn = lightzero_pong_eval_smoke_gpu
    elif compute == GPU_L4_T4_CPU8_COMPUTE:
        eval_fn = lightzero_pong_eval_smoke_gpu_cpu8
    elif compute == GPU_L4_T4_CPU40_COMPUTE:
        eval_fn = lightzero_pong_eval_smoke_gpu_cpu40
    else:
        raise ValueError(
            f"unknown compute: {compute!r}; expected 'cpu', 'gpu-l4-t4', "
            f"{GPU_L4_T4_CPU8_COMPUTE!r}, or {GPU_L4_T4_CPU40_COMPUTE!r}"
        )
    selected_refs = _selected_checkpoint_refs(
        checkpoint_ref=checkpoint_ref,
        checkpoint_refs=checkpoint_refs,
        checkpoint_ref_template=checkpoint_ref_template,
        selected_iterations=selected_iterations,
    )
    effective_eval_seed_count = (
        eval_seed_count
        if eval_seed_count is not None or eval_seeds is not None
        else DEFAULT_EVAL_SEED_COUNT
    )
    eval_seed_values, eval_seed_sampler_seed = _parse_eval_seeds(
        seed=seed,
        eval_seeds=eval_seeds,
        eval_seed_count=effective_eval_seed_count,
        eval_seed_rng_seed=eval_seed_rng_seed,
    )
    effective_max_eval_steps, effective_step_detail_limit = _detail_settings(
        eval_pass=eval_pass,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        low_detail_max_eval_steps=low_detail_max_eval_steps,
        low_detail_step_detail_limit=low_detail_step_detail_limit,
        high_detail_max_eval_steps=high_detail_max_eval_steps,
        high_detail_step_detail_limit=high_detail_step_detail_limit,
    )
    single_job = len(selected_refs) == 1 and len(eval_seed_values) == 1
    if not parallel and single_job and output_ref is not None:
        selected_output_refs = [output_ref]
    else:
        if output_ref is not None and not single_job:
            raise ValueError("output_ref is only valid for a single checkpoint/seed eval")
        stamp = runs.utc_stamp()
        clean_eval_id = _safe_generated_id(eval_id, fallback="eval")
        selected_output_refs = [
            _parallel_output_ref(
                run_id=run_id,
                attempt_id=attempt_id,
                eval_id=clean_eval_id,
                checkpoint_label=_checkpoint_label(ref, index=index),
                eval_pass=eval_pass,
                max_eval_steps=effective_max_eval_steps,
                seed=eval_seed,
                stamp=stamp,
            )
            for eval_seed in eval_seed_values
            for index, ref in enumerate(selected_refs)
        ]

    shared_kwargs = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "env_id": env_id,
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "update_per_collect": update_per_collect,
        "max_episode_steps": max_episode_steps,
        "game_segment_length": game_segment_length,
        "max_eval_steps": effective_max_eval_steps,
        "step_detail_limit": effective_step_detail_limit,
        "allow_model_fallback": allow_model_fallback,
        "run_stock_evaluator": run_stock_evaluator,
        "stock_only": stock_only,
        "skip_manual_rollout": effective_skip_manual_rollout,
        "slim_manifest": slim_manifest,
        "optimizer_phase_timing": optimizer_phase_timing,
    }
    remote_kwargs = {
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "update_per_collect": update_per_collect,
        "max_episode_steps": max_episode_steps,
        "game_segment_length": game_segment_length,
        "max_eval_steps": effective_max_eval_steps,
        "step_detail_limit": effective_step_detail_limit,
        "allow_model_fallback": allow_model_fallback,
        "run_stock_evaluator": run_stock_evaluator,
        "skip_manual_rollout": effective_skip_manual_rollout,
        "emit_result_json": not summary_only,
        "quiet_framework_logs": quiet_framework_logs,
        "optimizer_phase_timing": optimizer_phase_timing,
    }
    cap_warning = _eval_cap_warning(
        max_episode_steps=max_episode_steps,
        max_eval_steps=effective_max_eval_steps,
    )
    if cap_warning is not None:
        print(f"WARNING: {cap_warning['message']}")
    if parallel or len(selected_refs) > 1 or len(eval_seed_values) > 1:
        jobs = [
            {
                "index": job_index,
                "checkpoint_ref": ref,
                "checkpoint_label": _checkpoint_label(ref, index=checkpoint_index),
                "output_ref": selected_output_refs[job_index],
                "seed": eval_seed,
            }
            for job_index, (eval_seed, checkpoint_index, ref) in enumerate(
                (eval_seed, checkpoint_index, ref)
                for eval_seed in eval_seed_values
                for checkpoint_index, ref in enumerate(selected_refs)
            )
        ]
        starmap_inputs = [
            (
                job["checkpoint_ref"],
                job["output_ref"],
                run_id,
                attempt_id,
                env_id,
                job["seed"],
            )
            for job in jobs
        ]
        results = list(
            eval_fn.starmap(
                starmap_inputs,
                kwargs=remote_kwargs,
            )
        )
        table = [
            _parallel_table_row(job, result)
            for job, result in zip(jobs, results, strict=True)
        ]
        survival_aggregate_table = _survival_aggregate_table(table)
        survival_table = _survival_table(table)
        manifest_stamp = runs.utc_stamp()
        clean_eval_id = _safe_generated_id(eval_id, fallback="eval")
        manifest_ref = manifest_ref or (
            runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, clean_eval_id)
            / (
                f"manifest_{eval_pass}_steps{effective_max_eval_steps}"
                f"_seeds{_seed_panel_label(eval_seed_values)}_{manifest_stamp}.json"
            )
        ).as_posix()
        manifest = {
            "schema": "curvyzero_lightzero_visual_pong_parallel_eval/v0",
            "ok": all(bool(result.get("ok")) for result in results),
            "created_at": runs.utc_timestamp(),
            "job_kind": "lightzero_official_visual_pong_parallel_checkpoint_eval",
            "eval_id": clean_eval_id,
            "eval_pass": eval_pass,
            "retention_policy": DEFAULT_PARALLEL_RETENTION_POLICY,
            "artifact_naming": {
                "per_checkpoint": (
                    "attempt eval root / "
                    "{checkpoint_label}_{eval_pass}_steps{max_eval_steps}_seed{seed} / "
                    "lightzero_visual_pong_eval_{checkpoint_label}_{eval_pass}_steps"
                    "{max_eval_steps}_seed{seed}_{utc_stamp}.json"
                ),
                "manifest": "attempt eval root / manifest_{eval_pass}_steps{max_eval_steps}_seeds{seed_list}_{utc_stamp}.json",
            },
            "selection": {
                "checkpoint_ref": checkpoint_ref,
                "checkpoint_refs": checkpoint_refs,
                "checkpoint_ref_template": checkpoint_ref_template,
                "selected_iterations": selected_iterations,
                "eval_seeds": eval_seed_values,
                "eval_seed_count": effective_eval_seed_count,
                "eval_seed_sampler_seed": eval_seed_sampler_seed,
                "jobs": jobs,
            },
            "config": {
                **shared_kwargs,
                "seed": seed,
                "eval_seeds": eval_seed_values,
                "eval_seed_count": effective_eval_seed_count,
                "eval_seed_sampler_seed": eval_seed_sampler_seed,
            },
            "compute": compute,
            "warnings": [cap_warning] if cap_warning is not None else [],
            "table_fields": [
                "checkpoint",
                "seed",
                "stock_steps_survived",
                "steps_survived",
                "eval_cap_steps",
                "survival_fraction",
                "survived_to_cap",
                "ok",
                "strict_load",
                "stock_episode_length",
                "stock_return",
                "return",
                "stock_nonzero_reward_count",
                "stock_positive_reward_count",
                "stock_negative_reward_count",
                "stock_done",
                "stock_truncated",
                "stock_terminal_reason",
                "stock_action_histogram",
                "nonzero_reward_count",
                "positive_reward_count",
                "action_histogram",
                "dominant_action",
                "dominant_action_share",
                "action_entropy",
                "stock_manual_match",
                "stock_manual_status",
                "fallback_used",
                "elapsed_sec",
                "episode_elapsed_sec",
                "artifact_ref",
                "checkpoint_ref",
                "verdict",
            ],
            "survival_aggregate_table": survival_aggregate_table,
            "survival_table": survival_table,
            "table": table,
            "output_refs": [job["output_ref"] for job in jobs],
            "artifacts": [
                result.get("artifact")
                for result in results
                if isinstance(result, dict) and result.get("artifact")
            ],
        }
        if slim_manifest:
            manifest["results_omitted"] = True
            manifest["result_count"] = len(results)
        else:
            manifest["results"] = results
        manifest_result = lightzero_pong_eval_parallel_manifest.remote(
            manifest_ref=manifest_ref,
            manifest=manifest,
        )
        if summary_only:
            print("# aggregate_by_checkpoint")
            print(
                _tsv_table(
                    survival_aggregate_table,
                    [
                        "checkpoint",
                        "seeds",
                        "mean_steps",
                        "median_steps",
                        "min_steps",
                        "max_steps",
                        "ok",
                        "mean_score",
                    ],
                )
            )
            print("# per_checkpoint_seed_curve")
            print(
                _tsv_table(
                    survival_table,
                    [
                        "checkpoint",
                        "seed",
                        "stock_steps",
                        "manual_steps",
                        "cap",
                        "stock_return",
                        "stock_terminal",
                        "ok",
                        "strict",
                        "artifact_ref",
                    ],
                )
            )
            if eval_seed_sampler_seed is not None:
                print(f"eval_seed_sampler_seed\t{eval_seed_sampler_seed}")
                print("eval_seeds\t" + ",".join(str(item) for item in eval_seed_values))
            print(f"manifest_ref\t{manifest_ref}")
            return
        print(
            json.dumps(
                {
                    "survival_aggregate_table": survival_aggregate_table,
                    "survival_table": survival_table,
                    "table": table,
                    "manifest": manifest_result,
                    "manifest_ref": manifest_ref,
                    "jobs": jobs,
                    "output_refs": [job["output_ref"] for job in jobs],
                    "results": results,
                },
                indent=2,
            )
        )
        return

    result = eval_fn.remote(
        checkpoint_ref=selected_refs[0],
        output_ref=selected_output_refs[0],
        run_id=run_id,
        attempt_id=attempt_id,
        env_id=env_id,
        seed=eval_seed_values[0],
        **remote_kwargs,
    )
    if summary_only:
        summary = _single_summary(result)
        survival_table = _survival_table(summary["table"])
        survival_aggregate_table = _survival_aggregate_table(summary["table"])
        print("# aggregate_by_checkpoint")
        print(
            _tsv_table(
                survival_aggregate_table,
                [
                    "checkpoint",
                    "seeds",
                    "mean_steps",
                    "median_steps",
                    "min_steps",
                    "max_steps",
                    "ok",
                    "mean_score",
                ],
            )
        )
        print("# per_checkpoint_seed_curve")
        print(
            _tsv_table(
                survival_table,
                [
                    "checkpoint",
                    "seed",
                    "stock_steps",
                    "manual_steps",
                    "cap",
                    "stock_return",
                    "stock_terminal",
                    "ok",
                    "strict",
                    "artifact_ref",
                ],
            )
        )
        return
    else:
        print(json.dumps(result, indent=2))
