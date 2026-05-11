"""Tiny non-LightZero OpenSpiel AlphaZero control on Modal.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.openspiel_alphazero_quick_control

This invokes OpenSpiel's official AlphaZero implementation for a deliberately
small TicTacToe training run. It is a control for actor/replay/learner/checkpoint
plumbing, not a claim of solved play.
"""

from __future__ import annotations

import json
import shutil
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-openspiel-alphazero-quick-control"
TASK_ID = "openspiel-alphazero-tictactoe"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "open_spiel",
        "jax[cpu]",
        "flax",
        "optax",
        "chex",
        "absl-py",
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _version_or_missing(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def _scan_files(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": stat.st_size,
                "mtime": round(stat.st_mtime, 3),
            }
        )
    return files


def _read_jsonl_tail(path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"unparsed": line})
    return rows


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, cpu=4, memory=8192, timeout=900)
def run_control(
    run_id: str,
    attempt_id: str,
    *,
    game: str = "tic_tac_toe",
    max_steps: int = 1,
    actors: int = 1,
    evaluators: int = 0,
    max_simulations: int = 2,
    train_batch_size: int = 2,
    replay_buffer_size: int = 8,
    replay_buffer_reuse: int = 1,
    nn_model: str = "mlp",
    nn_width: int = 16,
    nn_depth: int = 1,
    eval_levels: int = 2,
) -> dict[str, Any]:
    import pyspiel
    from open_spiel.python.algorithms.alpha_zero import alpha_zero

    started_at = runs.utc_timestamp()
    run_ref = runs.run_root_ref(TASK_ID, run_id)
    attempt_ref = runs.attempt_root_ref(TASK_ID, run_id, attempt_id)
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    summary_ref = train_ref / "summary.json"
    local_path = Path("/tmp") / "curvyzero-openspiel-az" / run_id / attempt_id

    if local_path.exists():
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    config = alpha_zero.Config(
        game=game,
        path=str(local_path),
        learning_rate=1e-3,
        weight_decay=1e-4,
        decouple_weight_decay=False,
        train_batch_size=train_batch_size,
        replay_buffer_size=replay_buffer_size,
        replay_buffer_reuse=replay_buffer_reuse,
        max_steps=max_steps,
        checkpoint_freq=1,
        actors=actors,
        evaluators=evaluators,
        evaluation_window=4,
        eval_levels=eval_levels,
        uct_c=1.41,
        max_simulations=max_simulations,
        policy_alpha=1.0,
        policy_epsilon=0.25,
        temperature=1.0,
        temperature_drop=2,
        nn_model=nn_model,
        nn_width=nn_width,
        nn_depth=nn_depth,
        observation_shape=None,
        output_size=None,
        verbose=False,
        quiet=True,
        nn_api_version="linen",
    )

    game_obj = pyspiel.load_game(game)
    surface = {
        "game": game,
        "num_players": game_obj.num_players(),
        "reward_model": str(game_obj.get_type().reward_model),
        "dynamics": str(game_obj.get_type().dynamics),
        "max_game_length": game_obj.max_game_length(),
        "observation_tensor_shape": list(game_obj.observation_tensor_shape()),
        "num_distinct_actions": game_obj.num_distinct_actions(),
        "max_steps": max_steps,
        "actors": actors,
        "evaluators": evaluators,
        "max_simulations": max_simulations,
        "train_batch_size": train_batch_size,
        "replay_buffer_size": replay_buffer_size,
        "replay_buffer_reuse": replay_buffer_reuse,
        "nn_model": nn_model,
        "nn_width": nn_width,
        "nn_depth": nn_depth,
        "nn_api_version": "linen",
    }

    run_config = {
        "source": "OpenSpiel official Python AlphaZero implementation",
        "official_command_shape": (
            "python3 open_spiel/python/examples/alpha_zero.py "
            "--game tic_tac_toe --nn_model mlp"
        ),
        "surface": surface,
    }
    runs.write_json(
        runs.volume_path(RUNS_MOUNT, runs.run_manifest_ref(TASK_ID, run_id)),
        runs.run_manifest(task_id=TASK_ID, run_id=run_id, config=run_config, created_at=started_at),
    )
    runs.write_json(
        runs.volume_path(RUNS_MOUNT, runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id)),
        runs.attempt_manifest(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status="running",
            started_at=started_at,
            config=run_config,
        ),
    )

    t0 = time.perf_counter()
    error: str | None = None
    try:
        alpha_zero.alpha_zero(config)
    except Exception as exc:  # pragma: no cover - remote control diagnosis.
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0

    files = _scan_files(local_path)
    checkpoint_files = [item for item in files if item["path"].startswith("checkpoint-")]
    learner_tail = _read_jsonl_tail(local_path / "learner.jsonl")
    ok = error is None and bool(checkpoint_files) and bool(learner_tail)

    artifact_root = runs.volume_path(RUNS_MOUNT, train_ref / "openspiel_alpha_zero")
    if local_path.exists():
        _copy_tree(local_path, artifact_root)
    summary = {
        "ok": ok,
        "error": error,
        "started_at": started_at,
        "ended_at": runs.utc_timestamp(),
        "elapsed_sec": elapsed,
        "packages": {
            "open_spiel": _version_or_missing("open_spiel"),
            "jax": _version_or_missing("jax"),
            "jaxlib": _version_or_missing("jaxlib"),
            "flax": _version_or_missing("flax"),
            "optax": _version_or_missing("optax"),
            "chex": _version_or_missing("chex"),
            "numpy": _version_or_missing("numpy"),
        },
        "surface": surface,
        "artifact_ref": (train_ref / "openspiel_alpha_zero").as_posix(),
        "files": files,
        "checkpoint_files": checkpoint_files,
        "learner_tail": learner_tail,
    }
    runs.write_json(runs.volume_path(RUNS_MOUNT, summary_ref), summary)
    ended_at = summary["ended_at"]
    status = "completed" if ok else "failed"
    summary_ref_text = summary_ref.as_posix()
    runs.write_json(
        runs.volume_path(RUNS_MOUNT, runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id)),
        runs.attempt_manifest(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref_text,
            config=run_config,
        ),
    )
    runs.write_json(
        runs.volume_path(RUNS_MOUNT, runs.latest_attempt_ref(TASK_ID, run_id)),
        runs.latest_attempt_pointer(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref_text,
        ),
    )
    runs_volume.commit()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


@app.local_entrypoint()
def main(
    run_id: str = "openspiel-alphazero-tictactoe-20260511-s300",
    attempt_id: str = "train-tictactoe-1step-mlp-cpu",
    max_steps: int = 1,
    actors: int = 1,
    max_simulations: int = 2,
) -> None:
    summary = run_control.remote(
        run_id=run_id,
        attempt_id=attempt_id,
        max_steps=max_steps,
        actors=actors,
        max_simulations=max_simulations,
    )
    if not summary.get("ok"):
        raise SystemExit(f"OpenSpiel AlphaZero control failed: {summary.get('error')}")
