"""Modal smoke for LightZero PyTorch to JAX shadow-model parity.

This is not a trainer and does not touch live runs.  It proves the optimizer's
JAX shadow model can reproduce raw LightZero MuZero inference in a Modal image
that has both LightZero/Torch and JAX installed.

Run:

    uv run --extra modal modal run \
      -m curvyzero.infra.modal.lightzero_jax_shadow_model_parity \
      --compute l4 \
      --batch-size 2
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
import subprocess
from typing import Any

import modal
import numpy as np

from curvyzero.contracts.curvytron import curvytron_runs_volume_name
from curvyzero.contracts.curvytron import modal_volume_kwargs_for_name
from curvyzero.infra.modal.mctx_dependency_smoke import JAX_VERSION

APP_NAME = "curvyzero-lightzero-jax-shadow-model-parity"
REMOTE_ROOT = Path("/repo")
RUNS_MOUNT = Path("/runs")
LIGHTZERO_VERSION = "0.2.0"
TORCH_CUDA12_VERSION = "2.8.0"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"LightZero=={LIGHTZERO_VERSION}",
        f"jax[cuda12]=={JAX_VERSION}",
        f"torch=={TORCH_CUDA12_VERSION}",
        "numpy>=1.26",
    )
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name(
    curvytron_runs_volume_name(),
    **modal_volume_kwargs_for_name(curvytron_runs_volume_name()),
)


def _version_or_missing(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def _nvidia_smi() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _torch_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fresh_model_parity_smoke(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from lzero.model.muzero_model import MuZeroModel

    from curvyzero.training.lightzero_jax_shadow_model_parity import (
        compare_arrays,
        deterministic_observation_batch,
        jax_shadow_from_state_dict,
        profile_only_report_base,
    )

    batch_size = int(config.get("batch_size", 2))
    seed = int(config.get("seed", 0))
    atol = float(config.get("atol", 1e-4))
    rtol = float(config.get("rtol", 1e-4))
    require_gpu_backend = bool(config.get("require_gpu_backend", True))
    jax_platform = config.get("jax_platform")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MuZeroModel(
        observation_shape=[4, 64, 64],
        action_space_size=3,
        reward_support_size=3,
        value_support_size=3,
        self_supervised_learning_loss=True,
        downsample=True,
        norm_type="BN",
    ).to(device)
    _make_fresh_model_heads_nonzero(model, torch=torch, seed=seed + 1000)
    model.eval()

    shadow = jax_shadow_from_state_dict(model.state_dict(), platform=jax_platform)
    backend = shadow.jax.default_backend()
    problems: list[str] = []
    if require_gpu_backend and backend not in {"gpu", "cuda"}:
        problems.append(f"expected JAX GPU backend, got {backend!r}")

    comparisons: list[dict[str, Any]] = []
    for index, kind in enumerate(("zeros", "ramp", "random")):
        obs_np = deterministic_observation_batch(
            batch_size=batch_size,
            kind=kind,
            seed=seed + index,
        )
        obs_torch = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            torch_initial = model.initial_inference(obs_torch)
        jax_initial = shadow.initial_inference(obs_np)
        for output_name in ("policy_logits", "value", "reward", "latent_state"):
            comparisons.append(
                compare_arrays(
                    f"initial.{kind}.{output_name}",
                    _torch_to_numpy(getattr(torch_initial, output_name)),
                    np.asarray(jax_initial[output_name]),
                    atol=atol,
                    rtol=rtol,
                ).as_dict()
            )

        actions_np = np.arange(batch_size, dtype=np.int64) % shadow.action_space_size
        actions_torch = torch.as_tensor(actions_np, dtype=torch.long, device=device)
        with torch.no_grad():
            torch_recurrent = model.recurrent_inference(
                torch_initial.latent_state, actions_torch
            )
        jax_recurrent = shadow.recurrent_inference(
            jax_initial["latent_state"], actions_np
        )
        for output_name in ("policy_logits", "value", "reward", "latent_state"):
            comparisons.append(
                compare_arrays(
                    f"recurrent.{kind}.{output_name}",
                    _torch_to_numpy(getattr(torch_recurrent, output_name)),
                    np.asarray(jax_recurrent[output_name]),
                    atol=atol,
                    rtol=rtol,
                ).as_dict()
            )

    coverage = shadow.coverage_summary()
    bad = [item for item in comparisons if not bool(item["allclose"])]
    if bad:
        problems.append(f"{len(bad)} output comparisons failed tolerance")
    if not coverage["ok"]:
        problems.append("JAX shadow did not consume all required inference keys")

    report = {
        **profile_only_report_base(),
        "ok": not problems,
        "problems": problems,
        "packages": {
            "LightZero": _version_or_missing("LightZero"),
            "torch": _version_or_missing("torch"),
            "jax": _version_or_missing("jax"),
            "jaxlib": _version_or_missing("jaxlib"),
            "numpy": _version_or_missing("numpy"),
        },
        "jax": {
            "backend": backend,
            "devices": [str(device) for device in shadow.jax.devices()],
        },
        "torch": {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "nvidia_smi": _nvidia_smi(),
        "batch_size": batch_size,
        "seed": seed,
        "model_surface": {
            "model_class": "lzero.model.muzero_model.MuZeroModel",
            "observation_shape": [4, 64, 64],
            "action_space_size": int(shadow.action_space_size),
            "reward_support_size": int(shadow.reward_support_size),
            "value_support_size": int(shadow.value_support_size),
            "state_dict_key_count": len(model.state_dict()),
            "fresh_model_zero_initialized_heads_rewritten": True,
        },
        "coverage": coverage,
        "comparisons": comparisons,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _checkpoint_model_parity_smoke(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    from curvyzero.training.lightzero_checkpoint_opponent_provider import (
        load_lightzero_curvytron_visual_survival_policy,
    )
    from curvyzero.training.lightzero_jax_shadow_model_parity import (
        checkpoint_sha256,
        compare_arrays,
        deterministic_observation_batch,
        jax_shadow_from_state_dict,
        profile_only_report_base,
        require_immutable_checkpoint_ref,
    )

    checkpoint_ref = require_immutable_checkpoint_ref(str(config["checkpoint_ref"]))
    checkpoint_path = _runs_checkpoint_path(checkpoint_ref)
    batch_size = int(config.get("batch_size", 2))
    seed = int(config.get("seed", 0))
    num_simulations = int(config.get("num_simulations", 2))
    state_key = config.get("state_key") or None
    atol = float(config.get("atol", 5e-4))
    rtol = float(config.get("rtol", 5e-4))
    require_gpu_backend = bool(config.get("require_gpu_backend", True))
    use_cuda = bool(config.get("use_cuda", True))
    if not checkpoint_path.is_file():
        report = {
            **profile_only_report_base(),
            "ok": False,
            "problems": [f"checkpoint does not exist: {checkpoint_path}"],
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_path": str(checkpoint_path),
            "packages": {
                "LightZero": _version_or_missing("LightZero"),
                "torch": _version_or_missing("torch"),
                "jax": _version_or_missing("jax"),
                "jaxlib": _version_or_missing("jaxlib"),
                "numpy": _version_or_missing("numpy"),
            },
            "nvidia_smi": _nvidia_smi(),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    policy, device, load_summary = load_lightzero_curvytron_visual_survival_policy(
        checkpoint_path=checkpoint_path,
        seed=seed,
        num_simulations=num_simulations,
        batch_size=batch_size,
        use_cuda=use_cuda,
        state_key=state_key,
    )
    model = getattr(policy, "_model")
    model.eval()
    shadow = jax_shadow_from_state_dict(model.state_dict(), platform=config.get("jax_platform"))
    backend = shadow.jax.default_backend()
    problems: list[str] = []
    if require_gpu_backend and backend not in {"gpu", "cuda"}:
        problems.append(f"expected JAX GPU backend, got {backend!r}")

    comparisons: list[dict[str, Any]] = []
    for index, kind in enumerate(("zeros", "ramp", "random")):
        obs_np = deterministic_observation_batch(
            batch_size=batch_size,
            kind=kind,
            seed=seed + index,
        )
        obs_torch = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            torch_initial = model.initial_inference(obs_torch)
        jax_initial = shadow.initial_inference(obs_np)
        comparisons.extend(
            _compare_output_bundle(
                prefix=f"initial.{kind}",
                torch_output=torch_initial,
                jax_output=jax_initial,
                atol=atol,
                rtol=rtol,
                compare_arrays=compare_arrays,
            )
        )

        actions_np = np.arange(batch_size, dtype=np.int64) % shadow.action_space_size
        actions_torch = torch.as_tensor(actions_np, dtype=torch.long, device=device)
        with torch.no_grad():
            torch_recurrent = model.recurrent_inference(
                torch_initial.latent_state, actions_torch
            )
        jax_recurrent = shadow.recurrent_inference(
            jax_initial["latent_state"], actions_np
        )
        jax_recurrent_from_torch_latent = shadow.recurrent_inference(
            _torch_to_numpy(torch_initial.latent_state), actions_np
        )
        comparisons.extend(
            _compare_output_bundle(
                prefix=f"recurrent.{kind}",
                torch_output=torch_recurrent,
                jax_output=jax_recurrent,
                atol=atol,
                rtol=rtol,
                compare_arrays=compare_arrays,
            )
        )
        comparisons.extend(
            _compare_output_bundle(
                prefix=f"recurrent_from_torch_latent.{kind}",
                torch_output=torch_recurrent,
                jax_output=jax_recurrent_from_torch_latent,
                atol=atol,
                rtol=rtol,
                compare_arrays=compare_arrays,
            )
        )

    coverage = shadow.coverage_summary()
    bad = [item for item in comparisons if not bool(item["allclose"])]
    if bad:
        problems.append(f"{len(bad)} output comparisons failed tolerance")
    if not coverage["ok"]:
        problems.append("JAX shadow did not consume all required inference keys")

    report = {
        **profile_only_report_base(),
        "ok": not problems,
        "problems": problems,
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_load_summary": load_summary,
        "packages": {
            "LightZero": _version_or_missing("LightZero"),
            "torch": _version_or_missing("torch"),
            "jax": _version_or_missing("jax"),
            "jaxlib": _version_or_missing("jaxlib"),
            "numpy": _version_or_missing("numpy"),
        },
        "jax": {
            "backend": backend,
            "devices": [str(device) for device in shadow.jax.devices()],
        },
        "torch": {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "nvidia_smi": _nvidia_smi(),
        "batch_size": batch_size,
        "seed": seed,
        "num_simulations": num_simulations,
        "model_surface": {
            "model_class": type(model).__module__ + "." + type(model).__name__,
            "action_space_size": int(shadow.action_space_size),
            "reward_support_size": int(shadow.reward_support_size),
            "value_support_size": int(shadow.value_support_size),
            "state_dict_key_count": len(model.state_dict()),
        },
        "coverage": coverage,
        "comparisons": comparisons,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _runs_checkpoint_path(checkpoint_ref: str) -> Path:
    raw = Path(checkpoint_ref)
    if raw.is_absolute():
        resolved = raw.resolve()
        try:
            resolved.relative_to(RUNS_MOUNT)
        except ValueError as exc:
            raise ValueError(
                f"absolute checkpoint ref must live under {RUNS_MOUNT}: {checkpoint_ref!r}"
            ) from exc
        return resolved
    return (RUNS_MOUNT / raw).resolve()


def _compare_output_bundle(
    *,
    prefix: str,
    torch_output: Any,
    jax_output: dict[str, Any],
    atol: float,
    rtol: float,
    compare_arrays: Any,
) -> list[dict[str, Any]]:
    from curvyzero.training.lightzero_jax_shadow_model_parity import (
        inverse_scalar_transform_logits,
    )

    comparisons: list[dict[str, Any]] = []
    for output_name in ("policy_logits", "value", "reward", "latent_state"):
        torch_value = _torch_to_numpy(getattr(torch_output, output_name))
        jax_value = np.asarray(jax_output[output_name])
        comparisons.append(
            compare_arrays(
                f"{prefix}.{output_name}",
                torch_value,
                jax_value,
                atol=atol,
                rtol=rtol,
            ).as_dict()
        )
        if (
            output_name in {"value", "reward"}
            and torch_value.ndim == 2
            and jax_value.ndim == 2
        ):
            support_scale = (int(torch_value.shape[1]) - 1) // 2
            comparisons.append(
                compare_arrays(
                    f"{prefix}.{output_name}_scalar",
                    inverse_scalar_transform_logits(
                        torch_value,
                        support_scale=support_scale,
                    ),
                    inverse_scalar_transform_logits(
                        jax_value,
                        support_scale=support_scale,
                    ),
                    atol=atol,
                    rtol=rtol,
                ).as_dict()
            )
    return comparisons


def _make_fresh_model_heads_nonzero(model: Any, *, torch: Any, seed: int) -> None:
    """Make a fresh LightZero model test non-trivial.

    LightZero initializes the last reward/value/policy linear layers to zero by
    default.  That is fine for training, but a parity smoke with zero heads can
    falsely report perfect policy/value parity even if earlier hidden tensors
    drift.  This only mutates the local fresh smoke model.
    """

    torch.manual_seed(int(seed))
    heads = (
        model.dynamics_network.fc_reward_head[3],
        model.prediction_network.fc_value[3],
        model.prediction_network.fc_policy[3],
    )
    with torch.no_grad():
        for head in heads:
            head.weight.uniform_(-0.05, 0.05)
            head.bias.uniform_(-0.05, 0.05)


@app.function(image=image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_fresh_model_parity_smoke_l4(config: dict[str, Any]) -> dict[str, Any]:
    return _fresh_model_parity_smoke(config)


@app.function(image=image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_fresh_model_parity_smoke_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _fresh_model_parity_smoke(config)


@app.function(
    image=image,
    gpu=["L4", "T4"],
    timeout=20 * 60,
    cpu=4.0,
    volumes={str(RUNS_MOUNT): runs_volume},
)
def run_checkpoint_parity_smoke_l4(config: dict[str, Any]) -> dict[str, Any]:
    return _checkpoint_model_parity_smoke(config)


@app.function(
    image=image,
    gpu="H100",
    timeout=20 * 60,
    cpu=4.0,
    volumes={str(RUNS_MOUNT): runs_volume},
)
def run_checkpoint_parity_smoke_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _checkpoint_model_parity_smoke(config)


@app.local_entrypoint()
def main(
    compute: str = "l4",
    checkpoint_ref: str = "",
    state_key: str = "",
    batch_size: int = 2,
    num_simulations: int = 2,
    seed: int = 0,
    atol: float = 5e-4,
    rtol: float = 5e-4,
    require_gpu_backend: bool = True,
) -> None:
    config = {
        "batch_size": batch_size,
        "seed": seed,
        "num_simulations": num_simulations,
        "atol": atol,
        "rtol": rtol,
        "require_gpu_backend": require_gpu_backend,
    }
    if checkpoint_ref:
        config["checkpoint_ref"] = checkpoint_ref
        config["state_key"] = state_key
        config["use_cuda"] = True
        fn = run_checkpoint_parity_smoke_h100 if compute == "h100" else run_checkpoint_parity_smoke_l4
    else:
        fn = run_fresh_model_parity_smoke_h100 if compute == "h100" else run_fresh_model_parity_smoke_l4
    result = fn.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("ok"):
        raise SystemExit(1)
