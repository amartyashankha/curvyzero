"""Official MiniZero quick-run control on Modal.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.minizero_quick_run_control

This wrapper intentionally delegates training to MiniZero's official
``tools/quick-run.sh`` path. The default target is the smallest source-backed
board-game smoke: TicTacToe AlphaZero for one iteration.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-minizero-quick-run-control"
TASK_ID = "minizero-official-quick-run"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")
MINIZERO_SOURCE_LOCAL = Path("/private/tmp/minizero-main")
MINIZERO_ROOT = Path("/opt/minizero")

DEFAULT_RUN_ID = "minizero-tictactoe-az-quickrun-20260511-s0"
DEFAULT_ATTEMPT_ID = "train-tictactoe-az-iter1-l4t4"
DEFAULT_CONF_STR = (
    "actor_num_simulation=1:"
    "zero_num_parallel_games=1:"
    "zero_num_threads=1:"
    "zero_num_games_per_iteration=1:"
    "zero_display_latest_games=1:"
    "learner_training_step=1:"
    "learner_training_display_step=1:"
    "learner_batch_size=2:"
    "learner_num_thread=1:"
    "nn_num_hidden_channels=16:"
    "nn_num_value_hidden_channels=16:"
    "program_use_color_message=false"
)

if not MINIZERO_SOURCE_LOCAL.exists() and not MINIZERO_ROOT.exists():
    raise RuntimeError(
        f"MiniZero source checkout is required at {MINIZERO_SOURCE_LOCAL}; "
        "clone https://github.com/rlglab/minizero there before running this wrapper."
    )

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        add_python="3.11",
    )
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "libboost-all-dev",
        "libopencv-dev",
        "make",
        "pkg-config",
        "python3-dev",
        "python3-pybind11",
        "zlib1g-dev",
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        (
            "python -m pip install --index-url https://download.pytorch.org/whl/cu118 "
            "torch==2.1.2"
        ),
        "python -m pip install numpy pandas matplotlib",
        (
            "git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git "
            "/tmp/Arcade-Learning-Environment"
        ),
        (
            "cd /tmp/Arcade-Learning-Environment && "
            "git checkout d59d00688b58c5c14dff5fc79db5c22e86987f5d && "
            "mkdir build && cd build && "
            "cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 "
            "-DBUILD_PYTHON_LIB=OFF && "
            "cmake --build . --target install -j$(nproc) && "
            "ldconfig"
        ),
    )
    .add_local_dir(MINIZERO_SOURCE_LOCAL, remote_path=str(MINIZERO_ROOT), copy=True)
    .run_commands(
        (
            "cd /opt/minizero && "
            "if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
            "git init && git config user.email modal@curvyzero.local && "
            "git config user.name curvyzero-modal && git add . && "
            "git commit -m modal-source-snapshot; "
            "fi"
        )
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{MINIZERO_ROOT}",
            "CMAKE_PREFIX_PATH": (
                "/usr/local/lib/python3.11/site-packages/torch/share/cmake/Torch"
            ),
            "MPLBACKEND": "Agg",
        }
    )
    .add_local_dir(
        Path.cwd() / "src",
        remote_path=str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)

runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _version_or_missing(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def _run_text(command: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_sec": time.perf_counter() - started,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "elapsed_sec": time.perf_counter() - started,
            "timed_out": True,
        }


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


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _find_training_dir(workdir: Path, name: str) -> Path | None:
    explicit = workdir / name
    if explicit.exists():
        return explicit
    matches = sorted(
        (path for path in workdir.iterdir() if path.is_dir() and path.name.startswith(name)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _tail(path: Path, *, max_chars: int = 12000) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    gpu=["L4", "T4"],
    cpu=8,
    memory=32768,
    timeout=3600,
)
def run_control(
    run_id: str,
    attempt_id: str,
    *,
    game: str = "tictactoe",
    algorithm: str = "az",
    iterations: int = 1,
    conf_str: str = DEFAULT_CONF_STR,
    train_name: str | None = None,
    command_timeout_sec: int = 2400,
) -> dict[str, Any]:
    started_at = runs.utc_timestamp()
    train_name = train_name or f"{run_id}-{attempt_id}"
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    summary_ref = train_ref / "summary.json"
    artifact_ref = train_ref / "minizero"
    local_work = Path("/tmp") / "curvyzero-minizero" / run_id / attempt_id
    source_work = local_work / "minizero"

    if local_work.exists():
        shutil.rmtree(local_work)
    local_work.mkdir(parents=True, exist_ok=True)
    shutil.copytree(MINIZERO_ROOT, source_work, symlinks=True)

    config = {
        "source": "MiniZero official tools/quick-run.sh",
        "official_docs": {
            "training_doc": "docs/Training.md",
            "documented_tictactoe_command": "tools/quick-run.sh train tictactoe az 50",
        },
        "source_root": str(MINIZERO_ROOT),
        "game": game,
        "algorithm": algorithm,
        "iterations": iterations,
        "train_name": train_name,
        "conf_str": conf_str,
    }
    runs.write_json(
        runs.volume_path(RUNS_MOUNT, runs.run_manifest_ref(TASK_ID, run_id)),
        runs.run_manifest(task_id=TASK_ID, run_id=run_id, config=config, created_at=started_at),
    )
    runs.write_json(
        runs.volume_path(RUNS_MOUNT, runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id)),
        runs.attempt_manifest(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status="running",
            started_at=started_at,
            config=config,
        ),
    )

    env_report = _run_text(
        [
            "bash",
            "-lc",
            (
                "set -e; "
                "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; "
                "python - <<'PY'\n"
                "import torch\n"
                "print('torch', torch.__version__)\n"
                "print('cuda_available', torch.cuda.is_available())\n"
                "print('device_count', torch.cuda.device_count())\n"
                "PY"
            ),
        ],
        cwd=source_work,
        timeout=120,
    )

    command = [
        "bash",
        "tools/quick-run.sh",
        "train",
        game,
        algorithm,
        str(iterations),
        "-n",
        train_name,
        "-conf_str",
        conf_str,
    ]
    train_result = _run_text(command, cwd=source_work, timeout=command_timeout_sec)
    training_dir = _find_training_dir(source_work, train_name)

    artifact_root = runs.volume_path(RUNS_MOUNT, artifact_ref)
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    if training_dir is not None:
        _copy_if_exists(training_dir, artifact_root / training_dir.name)

    build_files = _scan_files(source_work / "build" / game)
    artifact_files = _scan_files(artifact_root)
    model_files = [item for item in artifact_files if item["path"].startswith(f"{train_name}/model/")]
    sgf_files = [item for item in artifact_files if item["path"].startswith(f"{train_name}/sgf/")]
    training_log = None if training_dir is None else _tail(training_dir / "Training.log")
    worker_log = None if training_dir is None else _tail(training_dir / "Worker.log")
    op_log = None if training_dir is None else _tail(training_dir / "op.log")

    ok = (
        train_result["returncode"] == 0
        and bool(model_files)
        and bool(sgf_files)
        and training_dir is not None
    )
    error = None
    if not ok:
        if train_result["timed_out"]:
            error = f"MiniZero quick-run timed out after {command_timeout_sec}s"
        else:
            error = f"MiniZero quick-run exited {train_result['returncode']}"

    summary = {
        "ok": ok,
        "error": error,
        "started_at": started_at,
        "ended_at": runs.utc_timestamp(),
        "packages": {
            "torch": _version_or_missing("torch"),
            "numpy": _version_or_missing("numpy"),
            "pandas": _version_or_missing("pandas"),
            "matplotlib": _version_or_missing("matplotlib"),
        },
        "surface": {
            "game": game,
            "algorithm": algorithm,
            "iterations": iterations,
            "conf_str": conf_str,
            "gpu": "L4/T4 Modal GPU via nvidia/cuda image",
        },
        "command": command,
        "env_report": env_report,
        "train_result": {
            key: value
            for key, value in train_result.items()
            if key not in {"stdout", "stderr"}
        },
        "stdout_tail": train_result["stdout"][-20000:],
        "stderr_tail": train_result["stderr"][-20000:],
        "training_dir": None if training_dir is None else str(training_dir),
        "artifact_ref": artifact_ref.as_posix(),
        "summary_ref": summary_ref.as_posix(),
        "build_files": build_files,
        "artifact_files": artifact_files,
        "model_files": model_files,
        "sgf_files": sgf_files,
        "training_log_tail": training_log,
        "worker_log_tail": worker_log,
        "op_log_tail": op_log,
        "next_command_if_blocked": (
            "uv run --extra modal modal run -m "
            "curvyzero.infra.modal.minizero_quick_run_control "
            "--run-id minizero-tictactoe-az-quickrun-20260511-s0b "
            "--attempt-id train-tictactoe-az-iter1-l4t4"
        ),
    }
    runs.write_json(runs.volume_path(RUNS_MOUNT, summary_ref), summary)
    status = "completed" if ok else "failed"
    ended_at = summary["ended_at"]
    runs.write_json(
        runs.volume_path(RUNS_MOUNT, runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id)),
        runs.attempt_manifest(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref.as_posix(),
            config=config,
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
            summary_ref=summary_ref.as_posix(),
        ),
    )
    runs_volume.commit()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


@app.local_entrypoint()
def main(
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    game: str = "tictactoe",
    algorithm: str = "az",
    iterations: int = 1,
    conf_str: str = DEFAULT_CONF_STR,
) -> None:
    summary = run_control.remote(
        run_id=run_id,
        attempt_id=attempt_id,
        game=game,
        algorithm=algorithm,
        iterations=iterations,
        conf_str=conf_str,
    )
    if not summary.get("ok"):
        raise SystemExit(summary.get("error") or "MiniZero quick-run failed")
