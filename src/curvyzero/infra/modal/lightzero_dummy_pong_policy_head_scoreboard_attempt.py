"""Modal wrapper for LightZero dummy Pong policy-head checkpoint scoreboards.

Run from the repository root:

    uv run --extra modal modal run \
      -m curvyzero.infra.modal.lightzero_dummy_pong_policy_head_scoreboard_attempt \
      --checkpoints lightzero:best=ref:training/lightzero-dummy-pong/RUN_ID/checkpoints/lightzero/ckpt_best.pth.tar \
      --episodes 20 \
      --seed 0 \
      --split-id dummy_pong_monitor_v0 \
      --split-role monitor

This scores direct policy-head greedy actions only: no MuZeroPolicy, no MCTS,
and no claim of parity with LightZero's evaluator.
"""

from __future__ import annotations

import json
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-lightzero-dummy-pong-policy-head-scoreboard"
TASK_ID = "lightzero-dummy-pong"
DEFAULT_EVAL_ID = "policy-head-scoreboard"
LIGHTZERO_VERSION = "0.2.0"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")
DEFAULT_FEATURE_MODE = "tabular_ego"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"LightZero=={LIGHTZERO_VERSION}", "numpy>=1.26")
    .env({"PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT}"})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(Path.cwd() / "scripts", remote_path=str(REMOTE_ROOT / "scripts"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _checkpoint_args_from_text(checkpoints: str) -> list[str]:
    values = [
        item.strip()
        for item in checkpoints.replace("\n", ",").split(",")
        if item.strip()
    ]
    if not values:
        raise ValueError(
            "checkpoints must include at least one lightzero:LABEL=PATH_OR_REF entry; "
            "use commas for multiple entries"
        )
    return values


def _parse_checkpoint_arg(value: str) -> tuple[str | None, str]:
    if not value.startswith("lightzero:"):
        raise ValueError(f"checkpoint must start with 'lightzero:': {value!r}")
    spec_text = value[len("lightzero:") :]
    if "=" not in spec_text:
        return None, spec_text
    label, path_text = spec_text.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("checkpoint label must not be empty")
    path_text = path_text.strip()
    if not path_text:
        raise ValueError("checkpoint path/ref must not be empty")
    return label, path_text


def _resolved_checkpoint_args(
    checkpoint_args: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    resolved_args = []
    inputs = []
    for value in checkpoint_args:
        label, path_text = _parse_checkpoint_arg(value)
        path, source = runs.resolve_mounted_ref_or_path(
            path_text,
            mount=RUNS_MOUNT,
            remote_root=REMOTE_ROOT,
        )
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {path}")
        resolved_arg = f"lightzero:{label}={path}" if label else f"lightzero:{path}"
        resolved_args.append(resolved_arg)
        inputs.append(
            {
                "checkpoint_arg": value,
                "checkpoint_label": label,
                "checkpoint_path_or_ref": path_text,
                "resolved_checkpoint_arg": resolved_arg,
                "resolved_checkpoint_path": str(path),
                **source,
                "file": runs.file_summary_any_mount(path, mount=RUNS_MOUNT),
            }
        )
    return resolved_args, inputs


def _eval_ref(
    *,
    output_ref: str | None,
    run_id: str | None,
    attempt_id: str | None,
    eval_id: str,
) -> PurePosixPath:
    explicit_ref = runs.explicit_volume_ref(output_ref or "") if output_ref else None
    if explicit_ref is not None:
        return explicit_ref
    if output_ref:
        return runs.require_relative_ref(output_ref)
    if run_id and attempt_id:
        return runs.attempt_eval_ref(
            TASK_ID,
            runs.clean_id(run_id, label="run_id"),
            runs.clean_id(attempt_id, label="attempt_id"),
            runs.clean_id(eval_id, label="eval_id"),
        )
    return runs.require_relative_ref(
        f"eval/{TASK_ID}/{runs.clean_id(eval_id, label='eval_id')}-{runs.utc_stamp()}"
    )


def _file_summaries_from_paths(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        name: runs.file_summary_any_mount(path, mount=RUNS_MOUNT)
        for name, path in sorted(paths.items())
        if path.is_file()
    }


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=20 * 60)
def score_lightzero_dummy_pong_policy_head(
    checkpoints: str,
    episodes: int = 20,
    seed: int = 0,
    split_id: str | None = "dummy_pong_monitor_v0",
    split_role: str | None = "monitor",
    run_id: str | None = None,
    attempt_id: str | None = None,
    eval_id: str = DEFAULT_EVAL_ID,
    output_ref: str | None = None,
    max_env_step: int = 64,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    paired_seats: bool = True,
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong_eval import run_dummy_pong_eval
    from scripts.run_dummy_pong_lightzero_checkpoint_scoreboard import (
        _as_lightzero_scoreboard_summary,
    )

    started = time.perf_counter()
    checkpoint_args = _checkpoint_args_from_text(checkpoints)
    resolved_checkpoint_args, checkpoint_inputs = _resolved_checkpoint_args(checkpoint_args)
    eval_ref = _eval_ref(
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=eval_id,
    )
    eval_dir = runs.volume_path(RUNS_MOUNT, eval_ref)
    summary = run_dummy_pong_eval(
        episodes=episodes,
        seed=seed,
        output_dir=eval_dir,
        lightzero_checkpoint_policies=resolved_checkpoint_args,
        lightzero_feature_mode=feature_mode,
        lightzero_max_env_step=max_env_step,
        paired_seats=paired_seats,
    )
    summary = _as_lightzero_scoreboard_summary(
        summary,
        checkpoint_args=checkpoint_args,
        output_dir=eval_dir,
        split_id=split_id,
        split_role=split_role,
        lightzero_env="dummy_pong_lag1",
        feature_mode=feature_mode,
        opponent_policy="random_uniform",
        max_env_step=max_env_step,
        paired_seats=paired_seats,
    )
    summary["modal_eval"] = {
        "schema": "curvyzero_modal_lightzero_policy_head_scoreboard/v0",
        "app_name": APP_NAME,
        "volume_name": VOLUME_NAME,
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "eval_id": eval_id,
        "eval_ref": eval_ref.as_posix(),
        "checkpoint_inputs": checkpoint_inputs,
        "plain_language": {
            "proves": "Independent LightZero checkpoint policy-head greedy scoring.",
            "does_not_prove": "MCTS behavior or LightZero evaluator parity.",
        },
    }
    summary_path = eval_dir / "summary.json"
    episodes_path = eval_dir / "episodes.jsonl"
    runs.write_json(summary_path, summary)
    runs_volume.commit()

    result = {
        "schema": "curvyzero_modal_lightzero_policy_head_scoreboard_result/v0",
        "app_name": APP_NAME,
        "volume_name": VOLUME_NAME,
        "output_refs": {
            "eval_dir": eval_ref.as_posix(),
            "summary_json": runs.file_ref(summary_path, mount=RUNS_MOUNT),
            "episodes_jsonl": runs.file_ref(episodes_path, mount=RUNS_MOUNT),
        },
        "file_summaries": {
            "checkpoint_inputs": checkpoint_inputs,
            "eval_outputs": _file_summaries_from_paths(
                {
                    "summary_json": summary_path,
                    "episodes_jsonl": episodes_path,
                }
            ),
        },
        "scoreboard": {
            "episodes_per_match": summary.get("episodes_per_match"),
            "total_episodes": summary.get("total_episodes"),
            "checkpoint_specs": summary.get("checkpoint_specs", []),
            "scoreboard_rows": summary.get("scoreboard_rows", []),
        },
        "committed": True,
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(
    checkpoints: str,
    episodes: int = 20,
    seed: int = 0,
    split_id: str | None = "dummy_pong_monitor_v0",
    split_role: str | None = "monitor",
    run_id: str | None = None,
    attempt_id: str | None = None,
    eval_id: str = DEFAULT_EVAL_ID,
    output_ref: str | None = None,
    max_env_step: int = 64,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    paired_seats: bool = True,
) -> None:
    started = time.perf_counter()
    result = score_lightzero_dummy_pong_policy_head.remote(
        checkpoints=checkpoints,
        episodes=episodes,
        seed=seed,
        split_id=split_id,
        split_role=split_role,
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=eval_id,
        output_ref=output_ref,
        max_env_step=max_env_step,
        feature_mode=feature_mode,
        paired_seats=paired_seats,
    )
    result["client_elapsed_sec"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
