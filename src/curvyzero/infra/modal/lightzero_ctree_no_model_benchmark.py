"""Modal runner for the no-model LightZero CTree boundary benchmark.

This is not a trainer and does not touch live Coach runs. It runs
``scripts/benchmark_lightzero_ctree_no_model.py`` inside a LightZero/Torch GPU
image so the `ctree-torch-d2h` backend measures a real CUDA-to-host boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import modal


APP_NAME = "curvyzero-lightzero-ctree-no-model-benchmark"
REMOTE_ROOT = Path("/repo")
LIGHTZERO_VERSION = "0.2.0"
TORCH_CUDA12_VERSION = "2.8.0"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"LightZero=={LIGHTZERO_VERSION}",
        f"torch=={TORCH_CUDA12_VERSION}",
        "numpy>=1.26",
        "Cython>=3",
    )
    .env({"PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'scripts'}"})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(
        Path.cwd() / "scripts",
        remote_path=str(REMOTE_ROOT / "scripts"),
        copy=True,
    )
    .run_commands(
        f"cd {REMOTE_ROOT} && python scripts/build_lightzero_ctree_a3.py build_ext --inplace"
    )
)

app = modal.App(APP_NAME)


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"expected positive comma-separated integers, got {raw!r}")
    return values


def _parse_csv_strings(raw: str, *, allowed: set[str]) -> list[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    bad = [value for value in values if value not in allowed]
    if not values or bad:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"expected comma-separated values from {expected}; got {raw!r}")
    return values


@app.function(image=image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_ctree_no_model_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _run_ctree_no_model(config)


@app.function(image=image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_ctree_no_model_l4_t4(config: dict[str, Any]) -> dict[str, Any]:
    return _run_ctree_no_model(config)


def _run_ctree_no_model(config: dict[str, Any]) -> dict[str, Any]:
    import sys

    for path in (str(REMOTE_ROOT / "scripts"), str(REMOTE_ROOT / "src")):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    import curvyzero

    repo_curvyzero = str(REMOTE_ROOT / "src" / "curvyzero")
    if repo_curvyzero not in curvyzero.__path__:
        curvyzero.__path__.insert(0, repo_curvyzero)
    for module_name in list(sys.modules):
        if module_name.startswith("curvyzero.vendor.lightzero_ctree_a3"):
            del sys.modules[module_name]

    from benchmark_lightzero_ctree_no_model import _run_case

    args = SimpleNamespace(
        iterations=int(config.get("iterations", 30)),
        warmup=int(config.get("warmup", 5)),
        seed=int(config.get("seed", 20260522)),
        pb_c_base=int(config.get("pb_c_base", 19652)),
        pb_c_init=float(config.get("pb_c_init", 1.25)),
        discount_factor=float(config.get("discount_factor", 0.997)),
        value_delta_max=float(config.get("value_delta_max", 0.01)),
        root_noise=str(config.get("root_noise", "zero")),
        root_noise_weight=float(config.get("root_noise_weight", 0.0)),
        root_dirichlet_alpha=float(config.get("root_dirichlet_alpha", 0.3)),
        torch_device=str(config.get("torch_device", "auto")),
        flat_a3_parity_check=bool(config.get("flat_a3_parity_check", False)),
    )
    roots = [int(value) for value in config["roots"]]
    simulations = [int(value) for value in config["simulations"]]
    backends = [str(value) for value in config["backends"]]
    legal_profiles = [str(value) for value in config["legal_profiles"]]
    rows = []
    for backend in backends:
        for root_count in roots:
            for num_simulations in simulations:
                for legal_profile in legal_profiles:
                    rows.append(
                        _run_case(
                            args,
                            backend=backend,
                            root_count=root_count,
                            simulations=num_simulations,
                            legal_profile=legal_profile,
                        )
                    )
    return {
        "schema_id": "curvyzero_lightzero_ctree_no_model_modal/v0",
        "lightzero_version": LIGHTZERO_VERSION,
        "torch_version": TORCH_CUDA12_VERSION,
        "row_count": len(rows),
        "rows": rows,
    }


@app.local_entrypoint()
def main(
    roots: str = "512,1024",
    simulations: str = "16",
    iterations: int = 30,
    warmup: int = 5,
    backends: str = "ctree-list,ctree-torch-d2h,fake-flat",
    legal_profiles: str = "all3",
    compute: str = "h100",
    flat_a3_parity_check: bool = False,
) -> None:
    config = {
        "roots": _parse_csv_ints(roots),
        "simulations": _parse_csv_ints(simulations),
        "iterations": int(iterations),
        "warmup": int(warmup),
        "backends": _parse_csv_strings(
            backends,
            allowed={"ctree-list", "ctree-torch-d2h", "ctree-flat-a3", "fake-flat"},
        ),
        "legal_profiles": _parse_csv_strings(
            legal_profiles,
            allowed={"all3", "single1", "mixed_2of3"},
        ),
        "torch_device": "auto",
        "flat_a3_parity_check": bool(flat_a3_parity_check),
    }
    if compute == "h100":
        result = run_ctree_no_model_h100.remote(config)
    elif compute == "l4-t4":
        result = run_ctree_no_model_l4_t4.remote(config)
    else:
        raise ValueError("compute must be 'h100' or 'l4-t4'")
    print(json.dumps(result, sort_keys=True))
