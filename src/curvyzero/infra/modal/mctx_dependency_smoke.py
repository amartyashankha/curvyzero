"""Contained Modal dependency smoke for the Mctx/JAX MuZero path.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.mctx_dependency_smoke --kind cpu

This is not a trainer. It only verifies that Modal can build an image with JAX
and Mctx, then execute one tiny synthetic Gumbel MuZero search.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import time
from importlib import metadata
from types import ModuleType

import modal

APP_NAME = "curvyzero-mctx-dependency-smoke"
MCTX_VERSION = "0.0.6"
JAX_VERSION = "0.7.0"

cpu_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    f"mctx=={MCTX_VERSION}",
    f"jax=={JAX_VERSION}",
    f"jaxlib=={JAX_VERSION}",
    "numpy>=1.26",
)

app = modal.App(APP_NAME)


def _version_or_missing(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def _nvidia_smi() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _import_modules(module_names: tuple[str, ...]) -> tuple[dict[str, str], dict[str, ModuleType]]:
    imports: dict[str, str] = {}
    modules: dict[str, ModuleType] = {}
    for module_name in module_names:
        try:
            modules[module_name] = importlib.import_module(module_name)
            imports[module_name] = "ok"
        except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
            imports[module_name] = f"{type(exc).__name__}: {exc}"
    return imports, modules


def _base_smoke_report(
    *,
    imports: dict[str, str],
    problems: list[str],
    batch_size: int,
    action_count: int,
    hidden_dim: int,
    num_simulations: int,
    max_depth: int,
) -> dict:
    return {
        "ok": False,
        "problems": problems,
        "packages": {
            "mctx": _version_or_missing("mctx"),
            "jax": _version_or_missing("jax"),
            "jaxlib": _version_or_missing("jaxlib"),
            "numpy": _version_or_missing("numpy"),
        },
        "imports": imports,
        "jax": None,
        "profile": {
            "batch_size": batch_size,
            "action_count": action_count,
            "hidden_dim": hidden_dim,
            "num_simulations": num_simulations,
            "max_depth": max_depth,
            "policy_kind": "gumbel_muzero_policy",
        },
        "timing": {
            "compile_plus_first_run_sec": None,
            "second_run_sec": None,
            "steady_state_second_run_sec": None,
            "decisions_per_sec_second_run": None,
            "simulations_per_sec_second_run": None,
        },
        "output": None,
        "nvidia_smi": _nvidia_smi(),
    }


def _run_mctx_search_smoke(
    *,
    batch_size: int = 4,
    action_count: int = 3,
    hidden_dim: int = 8,
    num_simulations: int = 4,
    max_depth: int = 4,
    require_gpu_backend: bool = False,
) -> dict:
    import functools

    if action_count != 3:
        raise ValueError(f"this smoke is fixed to CurvyZero A=3, got {action_count}")

    imports, modules = _import_modules(("jax", "jax.numpy", "mctx", "numpy"))
    problems = [
        f"failed to import {module_name}: {status}"
        for module_name, status in imports.items()
        if status != "ok"
    ]
    if problems:
        result = _base_smoke_report(
            imports=imports,
            problems=problems,
            batch_size=batch_size,
            action_count=action_count,
            hidden_dim=hidden_dim,
            num_simulations=num_simulations,
            max_depth=max_depth,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    jax = modules["jax"]
    jnp = modules["jax.numpy"]
    mctx = modules["mctx"]
    np = modules["numpy"]

    devices = jax.devices()
    backend = jax.default_backend()
    if require_gpu_backend and backend not in {"gpu", "cuda"}:
        problems.append(f"expected a GPU JAX backend, got {backend!r}")

    params = {
        "action_proj": jnp.array(
            [
                [-0.10, 0.00, 0.05, 0.10, -0.05, 0.02, 0.00, 0.03],
                [0.00, 0.04, -0.02, 0.00, 0.06, -0.03, 0.05, -0.01],
                [0.10, -0.02, 0.03, -0.08, 0.02, 0.05, -0.04, 0.00],
            ],
            dtype=jnp.float32,
        ),
        "policy_w": jnp.array(
            [
                [0.10, -0.04, 0.02],
                [-0.03, 0.07, 0.01],
                [0.05, 0.00, -0.05],
                [0.02, 0.03, 0.04],
                [-0.06, 0.02, 0.05],
                [0.01, -0.08, 0.03],
                [0.04, 0.04, -0.02],
                [-0.02, 0.01, 0.06],
            ],
            dtype=jnp.float32,
        ),
        "policy_b": jnp.array([0.00, 0.03, -0.01], dtype=jnp.float32),
        "value_w": jnp.array(
            [0.05, -0.02, 0.03, 0.04, -0.01, 0.02, 0.01, -0.03],
            dtype=jnp.float32,
        ),
    }

    embedding = jnp.linspace(-0.5, 0.5, batch_size * hidden_dim, dtype=jnp.float32).reshape(
        batch_size, hidden_dim
    )
    prior_logits = embedding @ params["policy_w"] + params["policy_b"]
    value = jnp.tanh(embedding @ params["value_w"])
    root = mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=embedding)
    invalid_actions = jnp.zeros((batch_size, action_count), dtype=jnp.bool_)

    def recurrent_fn(params, rng_key, action, embedding):
        del rng_key
        action_features = jax.nn.one_hot(action, action_count, dtype=jnp.float32)
        next_embedding = jnp.tanh(embedding + action_features @ params["action_proj"])
        prior_logits = next_embedding @ params["policy_w"] + params["policy_b"]
        value = jnp.tanh(next_embedding @ params["value_w"])
        reward = 0.01 * (action.astype(jnp.float32) - 1.0)
        discount = jnp.full_like(value, 0.99)
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=value,
            ),
            next_embedding,
        )

    @functools.partial(jax.jit, static_argnames=("num_simulations", "max_depth"))
    def run_search(params, rng_key, root, invalid_actions, *, num_simulations: int, max_depth: int):
        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=invalid_actions,
            max_depth=max_depth,
            max_num_considered_actions=action_count,
            gumbel_scale=1.0,
        )

    first_started = time.perf_counter()
    first_output = run_search(
        params,
        jax.random.PRNGKey(0),
        root,
        invalid_actions,
        num_simulations=num_simulations,
        max_depth=max_depth,
    )
    first_output.action_weights.block_until_ready()
    first_elapsed = time.perf_counter() - first_started

    second_started = time.perf_counter()
    output = run_search(
        params,
        jax.random.PRNGKey(1),
        root,
        invalid_actions,
        num_simulations=num_simulations,
        max_depth=max_depth,
    )
    output.action_weights.block_until_ready()
    second_elapsed = time.perf_counter() - second_started

    actions = np.asarray(output.action)
    action_weights = np.asarray(output.action_weights)
    row_sums = action_weights.sum(axis=1)
    finite_weights = bool(np.isfinite(action_weights).all())
    row_sums_close = bool(np.allclose(row_sums, 1.0, atol=1e-5))
    if not finite_weights:
        problems.append("action_weights contains non-finite values")
    if not row_sums_close:
        problems.append("action_weights rows do not sum to 1 within atol=1e-5")

    result = {
        "ok": not problems,
        "problems": problems,
        "packages": {
            "mctx": _version_or_missing("mctx"),
            "jax": _version_or_missing("jax"),
            "jaxlib": _version_or_missing("jaxlib"),
            "numpy": _version_or_missing("numpy"),
        },
        "imports": imports,
        "jax": {
            "default_backend": backend,
            "devices": [str(device) for device in devices],
            "device_count": len(devices),
        },
        "profile": {
            "batch_size": batch_size,
            "action_count": action_count,
            "hidden_dim": hidden_dim,
            "num_simulations": num_simulations,
            "max_depth": max_depth,
            "policy_kind": "gumbel_muzero_policy",
        },
        "timing": {
            "compile_plus_first_run_sec": first_elapsed,
            "second_run_sec": second_elapsed,
            "steady_state_second_run_sec": second_elapsed,
            "decisions_per_sec_second_run": batch_size / second_elapsed,
            "simulations_per_sec_second_run": (batch_size * num_simulations) / second_elapsed,
        },
        "output": {
            "actions": actions.astype(int).tolist(),
            "action_histogram": np.bincount(actions, minlength=action_count).astype(int).tolist(),
            "action_weights_finite": finite_weights,
            "action_weight_row_sums": row_sums.astype(float).tolist(),
        },
        "nvidia_smi": _nvidia_smi(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.function(image=cpu_image, timeout=5 * 60)
def cpu_smoke() -> dict:
    return _run_mctx_search_smoke(require_gpu_backend=False)


@app.local_entrypoint()
def main(kind: str = "cpu") -> None:
    if kind == "cpu":
        result = cpu_smoke.remote()
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        raise ValueError(f"unknown smoke kind: {kind}")
