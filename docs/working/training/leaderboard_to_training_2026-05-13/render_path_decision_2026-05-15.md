# Render Path Decision: 2026-05-15

## Current Answer

The old `curvy-v2refresh18p-20260514b` manifest was **not** using a GPU
observation renderer. Treat it as historical/diagnostic, not current launch
guidance.

It used:

```text
compute: gpu-h100-cpu40
source_state_trail_render_mode: body_circles_fast
source_state_bonus_render_mode: simple_symbols
```

Meaning, historically:

- H100 is used for LightZero policy/search/learner compute.
- `body_circles_fast + simple_symbols` was the fast CPU observation path wired
  into the stock trainer.
- `simple_symbols` is the implemented bonus-symbol approximation for the direct
  fast gray64 path. It is not proof that the whole observation renderer is on
  GPU.

## Evidence

- The old local manifest has all 18 rows set to `body_circles_fast` and
  `simple_symbols`; keep that as historical/current CPU evidence only.
- `docs/working/optimizer/coach_handoff_fast_stock_recommendation_2026-05-14.md`
  historically recommended H100 compute with
  `body_circles_fast + simple_symbols`. That recommendation is superseded here;
  it remains evidence that H100 compute is different from GPU observation
  rendering.
- `docs/working/optimizer/active_working_memory_2026-05-14.md` says the bonus
  symbol renderer is wired as an explicit stock training option, while the GPU
  browser-lines renderer is promising but not trusted.
- The actual training env calls CPU/Numpy render functions:
  `render_source_state_gray64_fast_player_perspectives`,
  `render_source_state_canvas_gray64_player_perspectives`, and
  `render_source_state_canvas_gray64`.
- The GPU render files are standalone probes/benchmarks:
  `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py` and
  `src/curvyzero/infra/modal/curvytron_gpu_render_probe.py`. Their docstrings
  say they do not touch trainers or Modal Volumes.

## Current Direction

The target policy observation surface is `browser_lines + simple_symbols`.
The current reliable backend is CPU `cpu_oracle`. The measured scalar
`jax_gpu` backend now reaches stock `train_muzero`, but it is slower than CPU
and fails in subprocess workers. The production backend stays `cpu_oracle`;
GPU `browser_lines + simple_symbols` is lab/profiling-only until
trainer-visible contract parity passes. `body_circles_fast + simple_symbols`
is historical CPU fallback, ablation, or control evidence only, and fresh
source-state trainer launches should reject it as a policy surface.

Moving this into the trainer efficiently is real feature work. It likely needs:

- a trainer env flag naming the observation backend;
- live `VectorMultiplayerEnv` state to GPU tensor conversion;
- exact or explicitly approximate bonus/trail semantics;
- batching across many env rows instead of one JAX call per scalar env step;
- parity tests against the CPU reference or an explicit approximation contract;
- a full-loop profile proving the overhead is actually lower.

A profile-only batched observation facade now exists for measurement. It should
remain outside live runs and stock LightZero defaults until the contract gates
pass.

Measured scalar canary:

- `cpu_oracle`, H100/base/C1/sim2/512: `15.54s`, `32.94` steps/s,
  `4.42ms` observation/step.
- scalar `jax_gpu`, same shape: `63.73s`, `8.03` steps/s, `80.31ms`
  observation/step.
- subprocess scalar `jax_gpu`, C2/sim2/32: failed before collection with JAX
  CUDA initialization errors in env workers.

Measured isolated batched H100 renderer:

- B64: `34.54ms` end-to-end, about `1853` frames/s, max diff `2`,
  mismatch fraction about `0.49%` on checked rows.
- B256: `104.78ms` end-to-end, about `2443` frames/s, max diff `2`,
  mismatch fraction about `0.56%` on checked rows.

## Current Stance

Do not silently call `body_circles_fast + simple_symbols` "GPU rendering." It is
CPU fast rendering running inside an H100 training container.

The optimizer direction is batched GPU renderer work, with CPU
browser-lines+symbols as the current reliable backend and comparison target.
New launch guidance must not copy the old body-circles CPU-fast commands unless
explicitly labeled as historical CPU fallback, ablation, or control.
