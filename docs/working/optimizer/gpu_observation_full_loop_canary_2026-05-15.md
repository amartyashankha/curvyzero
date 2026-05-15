# GPU Observation Full-Loop Canary

Date: 2026-05-15

Purpose: answer whether the CurvyTron policy observation renderer is actually
running on GPU in the canonical stock LightZero path, and whether that is faster.

## Plain Answer

`compute=gpu-*` was not enough. That only put the LightZero model/search/learner
on GPU. The CurvyTron observation renderer stayed on CPU unless the run passed:

```text
--policy-observation-backend jax_gpu
```

That flag is now wired through the canonical launcher into the stock
`lzero.entry.train_muzero` env config. The path runs, but the current
implementation is a scalar JAX call inside each env step. That shape is slower
than the CPU oracle. The promising GPU renderer is the isolated batched renderer,
not the scalar env-wrapper hook.

## Canonical Path Tested

All full-loop rows used:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode profile
--env-variant source_state_fixed_opponent
--source-state-trail-render-mode browser_lines
--source-state-bonus-render-mode simple_symbols
--opponent-runtime-mode blank_canvas_noop
--disable-death-for-profile
--env-manager-type base
--collector-env-num 1
--num-simulations 2
--stop-after-learner-train-calls 1
--no-background-eval-enabled
--no-background-gif-enabled
```

These rows are not learning claims. They are speed/plumbing profiles.

## Full-Loop Results

| run | backend | steps | wall | steps/s | obs mean | collector | policy collect | MCTS | learner | GPU memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cpu-oracle-c1-sim2-steps64-fresh` | `cpu_oracle` | 64 | `11.65s` | `5.49` | `8.64ms` | `3.99s` | `3.11s` | `0.37s` | `2.56s` | `1.1GB` |
| `jaxgpu-scalar-c1-sim2-steps64-fresh` | `jax_gpu` | 64 | `19.82s` | `3.23` | `79.44ms` | `8.41s` | `3.01s` | `0.27s` | `1.29s` | `62.0GB` |
| `cpu-oracle-c1-sim2-steps512` | `cpu_oracle` | 512 | `15.54s` | `32.94` | `4.42ms` | `9.57s` | `5.22s` | `2.50s` | `1.60s` | `1.1GB` |
| `jaxgpu-scalar-c1-sim2-steps512` | `jax_gpu` | 512 | `63.73s` | `8.03` | `80.31ms` | `49.01s` | `5.43s` | `2.55s` | `2.72s` | `62.0GB` |
| `opt-gpuobs-prefix-cpu-h100-s20260515` | `cpu_oracle` | 512 | `15.61s` | `32.80` | `4.50ms` | `10.18s` | `5.73s` | `2.42s` | `1.29s` | `1.1GB` |
| `opt-gpuobs-prefix-jax-h100-s20260515` | `jax_gpu` active-prefix bucket | 512 | `34.46s` | `14.86` | `24.39ms` | `20.52s` | `5.73s` | `2.51s` | `1.39s` | `61.9GB` |

Plain read:

- The GPU observation flag is real. It reached `train_muzero`, returned valid
  observations, and collected/learned.
- The scalar JAX-GPU backend is not a production speedup. It is about `4.1x`
  slower than CPU at 512 steps.
- The slowdown is in observation time, not search. At 512 steps, MCTS is nearly
  identical (`2.50s` CPU versus `2.55s` scalar GPU).
- The scalar JAX-GPU path also consumes about `62GB` on an H100 in this shape.
  That is a strong warning against subprocess env workers with one JAX renderer
  per worker.
- Active-prefix bucketing is a real improvement, but it is still not enough:
  the 512-step full-loop canary improved from `~80ms` observation mean to
  `24.39ms`, while the matched CPU oracle is `4.50ms`.

## Isolated Batched Renderer Results

The isolated benchmark uses the same target semantic surface:

```text
browser_lines + simple_symbols
block_704_gray64
real_env_rollout
H100
trail_slots=256
bonus_count=8
transfer_output=true
```

| batch | end-to-end median | frames/s | device render | host->device | device->host | parity |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `8.35ms` | `119.8` | `2.74ms` | `5.22ms` | `0.50ms` | max diff 1, `0.61%` pixels off |
| 64 | `34.54ms` | `1852.8` | `30.90ms` | `3.16ms` | `0.50ms` | max diff 2, `0.49%` pixels off |
| 256 | `104.78ms` | `2443.2` | `100.36ms` | `3.86ms` | `0.58ms` | max diff 2, `0.56%` pixels off |

Plain read:

- Batching is the win. Batch 1 is dominated by host transfer and launch overhead.
- Batch 256 is about `20x` more frames/sec than batch 1.
- The parity gap is tiny in the checked rows, but not exact. It looks like
  rounding/downsample edge differences, not missing objects.

## Why Scalar GPU Lost

The current `jax_gpu` trainer hook does this on every env step:

1. Convert one CPU env state into compact JAX inputs.
2. Copy that one row to the GPU.
3. Render player 0.
4. Render player 1.
5. Copy both frames back to NumPy.
6. Update the LightZero stack on CPU.

That pays GPU launch/copy overhead per env step and keeps the stock LightZero env
API as host NumPy. It proves plumbing, but it is the wrong performance shape.

## Current Decision

Do not make scalar `jax_gpu` the default training backend.

Use `cpu_oracle` for stock LightZero training until a batched GPU observation
backend exists. Keep the semantic surface fixed:

```text
browser_lines + simple_symbols
```

The next optimizer target is:

```text
batched GPU observation at the vector/env-manager/collector boundary
```

or a dedicated render service that receives many env states at once. Do not hide
this behind `compute=gpu-*`; every profile must print `policy_observation_backend`.

## Subprocess Canary

Tiny subprocess row:

```text
jaxgpu-subproc-c2-sim2-steps32
env_manager_type=subprocess
collector_env_num=2
```

Result: failed before collection. JAX reported CUDA initialization errors inside
the subprocess env workers and then fell back toward CPU; LightZero failed with
`RuntimeError: Exception in thread(...)`.

Plain read: do not use scalar `jax_gpu` with subprocess env workers. It is not
just slow; the process topology itself is unsafe for this backend.

## Hardware Sidecar

Matched isolated batch-256 rows, real-env rollout, `trail_slots=256`:

| GPU | end-to-end median | frames/s | device render |
| --- | ---: | ---: | ---: |
| H100 | `104.78ms` | `2443.2` | `100.36ms` |
| L4 | `1164.68ms` | `219.8` | `1159.49ms` |

Plain read: if we do build a batched GPU observation backend, H100 is the
serious renderer target for heavy trail histories. L4 is usable for development
but much slower in this particular block renderer.

## Next Experiments

Already launched or queued:

- `jaxgpu-subproc-c2-sim2-steps32`: failed before collection with JAX CUDA
  initialization errors in env workers.
- `gpu-l4-t4` batch-256 isolated renderer row: completed, but was about `11.1x`
  slower than H100 by end-to-end frames/sec.

Recommended next wave if more profiling is needed:

| row type | backend | collectors | cap | sims | env manager | reason |
| --- | --- | ---: | ---: | ---: | --- | --- |
| horizon | `cpu_oracle` | 1 | 2000 | 2 | base | longer CPU baseline |
| horizon | `jax_gpu` | 1 | 2000 | 2 | base | only if investigating scalar warmup/horizon effects; not a candidate training backend |
| scaling | `cpu_oracle` | 32/64 | 512 | 8 | subprocess | real stock loop throughput |
| search | `cpu_oracle` | 32/64 | 512 | 16 | subprocess | see if MCTS becomes dominant |
| render sidecar | batched JAX | 64/256/512 | n/a | n/a | n/a | design target for real GPU backend |

Do not scale scalar `jax_gpu` wide. The subprocess canary failed.
