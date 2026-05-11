"""Modal smoke tests and benchmarks for CurvyZero.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind tests
    uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind benchmark --episodes 1000

The image intentionally installs the project in editable form inside Modal so
remote tests exercise the same package structure as local development.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "curvyzero-smoke"
REMOTE_ROOT = Path("/repo")


def _ignore_modal_upload(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"})


runtime_base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
)

runtime_image = (
    runtime_base_image
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(Path.cwd() / "scripts", remote_path=str(REMOTE_ROOT / "scripts"), copy=True)
    .add_local_file(Path.cwd() / "pyproject.toml", remote_path=str(REMOTE_ROOT / "pyproject.toml"))
    .add_local_file(Path.cwd() / "README.md", remote_path=str(REMOTE_ROOT / "README.md"))
)

test_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26", "pytest>=8", "ruff>=0.6")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(Path.cwd() / "scripts", remote_path=str(REMOTE_ROOT / "scripts"), copy=True)
    .add_local_dir(Path.cwd() / "tests", remote_path=str(REMOTE_ROOT / "tests"), copy=True)
    .add_local_file(Path.cwd() / "pyproject.toml", remote_path=str(REMOTE_ROOT / "pyproject.toml"))
    .add_local_file(Path.cwd() / "README.md", remote_path=str(REMOTE_ROOT / "README.md"))
)

app = modal.App(APP_NAME)


@app.function(image=test_image, timeout=10 * 60)
def run_tests() -> str:
    completed = subprocess.run(
        ["python", "-m", "pytest", "-q"],
        cwd=REMOTE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    print(completed.stdout)
    return completed.stdout


@app.function(image=runtime_image, timeout=10 * 60)
def benchmark_env(episodes: int = 1_000, max_steps: int = 2_000, seed: int = 0) -> dict:
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")

    from curvyzero.env import CurvyTronConfig, CurvyTronEnv

    import numpy as np

    rng = np.random.default_rng(seed)
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    total_steps = 0
    started = time.perf_counter()

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        for _ in range(max_steps):
            result = env.step(
                {
                    "player_0": int(rng.integers(env.config.action_count)),
                    "player_1": int(rng.integers(env.config.action_count)),
                }
            )
            total_steps += 1
            if result.terminated["player_0"] or result.truncated["player_0"]:
                break

    elapsed = time.perf_counter() - started
    metrics = {
        "episodes": episodes,
        "steps": total_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": total_steps / elapsed,
        "episodes_per_sec": episodes / elapsed,
        "rules_hash": env.config.rules_hash,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


@app.function(image=runtime_image, gpu=["L4", "T4"], timeout=10 * 60)
def gpu_smoke() -> dict:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    info = {"nvidia_smi": completed.stdout.strip()}
    print(json.dumps(info, indent=2, sort_keys=True))
    return info


@app.local_entrypoint()
def main(kind: str = "tests", episodes: int = 1_000, max_steps: int = 2_000) -> None:
    if kind == "tests":
        run_tests.remote()
    elif kind == "benchmark":
        benchmark_env.remote(episodes=episodes, max_steps=max_steps)
    elif kind == "gpu":
        gpu_smoke.remote()
    else:
        raise ValueError(f"unknown smoke kind: {kind}")
