# 2026-05-09 Modal Mctx Dependency Smoke

## Question

Can Modal build and run the Mctx/JAX runtime we would need for the Mctx-first
MuZero lane?

## Setup

Two contained Modal modules:

- CPU: `curvyzero.infra.modal.mctx_dependency_smoke`
- GPU: `curvyzero.infra.modal.mctx_gpu_dependency_smoke`

Both run one tiny synthetic `gumbel_muzero_policy` search with `B=4`, `A=3`,
hidden dim 8, 4 simulations, and max depth 4. This is not training and not
Pong.

## Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_dependency_smoke --kind cpu
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke
```

## Results

CPU passed:

- `ok: true`
- JAX backend: `cpu`
- packages: `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6`
- finite action weights, row sums exactly `1.0`
- first run including compile: about `2.11s`
- second run: about `0.0012s`

GPU passed:

- `ok: true`
- JAX backend: `gpu`
- device: `cuda:0`
- `nvidia-smi`: `NVIDIA L4, 23034 MiB, 580.95.05`
- packages: `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6`
- finite action weights, row sums near `1.0`
- first run including compile: about `4.26s`
- second run: about `0.0022s`

## Interpretation

Modal GPU, JAX CUDA, and Mctx are viable for the next synthetic MuZero/search
benchmark. This does not prove Pong learning, CurvyTron learning, or a full
MuZero trainer. It only clears the dependency/runtime gate.

The first version of the module registered CPU and GPU functions together, so
the CPU-only run also built the CUDA image. That was fixed by splitting the GPU
smoke into `mctx_gpu_dependency_smoke.py`.

## Artifacts

Modal app runs:

- CPU: `ap-GZHlT9pGWpnpkI7tukoCLg`
- GPU: `ap-Gsju0I7DgJsoun5teJMh2h`

Changed modules:

- `src/curvyzero/infra/modal/mctx_dependency_smoke.py`
- `src/curvyzero/infra/modal/mctx_gpu_dependency_smoke.py`

## Follow-ups

- Build the Mctx synthetic benchmark on Modal GPU.
- Keep the benchmark separate from dependency smoke; do not raise batch size or
  simulations inside this smoke.
- Continue the stock-example lane separately with a LightZero import/config
  smoke before any CartPole or Atari training command.
