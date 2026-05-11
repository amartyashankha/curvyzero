# Observation Optimization Research

Date: 2026-05-10

Status: concise literature/search note for batched 2D ray/LiDAR observation
optimization. This is optimizer-owned speed guidance only.

## Search Terms

- `batched 2D ray circle intersection numpy numba`
- `LiDAR observation reinforcement learning vectorized ray casting`
- `Box2D dynamic tree ray cast broad phase`
- `uniform grid spatial hash broad phase collision detection CUDA`
- `Numba nopython prange performance tips`
- `JAX vmap scan batched environment rollout`
- `EnvPool highly parallel reinforcement learning environment execution`
- `Brax gymnax Mctx JAX accelerator RL`

## Verdict

Keep the near-term path CPU and fidelity-first:

1. Dense batched ray-circle math first.
2. Numba `njit`/`parallel=True` only after the dense NumPy/scalar-parity path is
   stable.
3. Uniform grid/spatial hash only if body/trail counts make dense all-circles
   tests scale poorly.
4. GPU/JAX only when env, observation, policy/search, and rollout loop stay on
   device long enough to amortize rewrite and transfer costs.

This matches current CurvyTron evidence: observation/ray work is still the
largest measured bucket, but source fidelity is not settled enough to justify a
new GPU runtime as the first move.

## Evidence Read

- Box2D's dynamic tree is the right conceptual broad-phase reference for many
  moving shapes: it organizes AABBs, supports queries/ray casts, and reports
  node/leaf visits. Its ray cast still relies on exact shape tests in a callback,
  with performance described as roughly `k * log(n)` for `k` hits and `n`
  proxies. Source: [Box2D dynamic tree docs](https://box2d.org/documentation/group__tree.html).
- For this repo's current ray-circle shape, dense batched NumPy is the simplest
  first bet: precompute ray origins/directions, arrange body centers/radii as
  contiguous arrays, compute all ray-circle intersections for `[B,P,R,N]`, then
  reduce nearest positive hit. This avoids Python callbacks and keeps parity
  easy.
- Numba is a good second rung when loops remain clearer than broadcasting or
  memory pressure gets high. Its own guidance says to profile real data, prefer
  no-python mode, and use `prange` with `parallel=True` for embarrassingly
  parallel loops. Source: [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html).
- Uniform grids/spatial hashes are a broad-phase optimization, not a fidelity
  shortcut. GPU Gems describes broad phase as conservative pruning before exact
  narrow phase, and spatial subdivision as uniform-grid partitioning that tests
  objects only when they share relevant cells. Source: [GPU Gems 3, chapter 32](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda).
- JAX becomes attractive only for an owned tensor runtime. `vmap` creates a
  batched function over array axes, and `lax.scan` lowers fixed-shape loops to a
  single WhileOp, but all loop-carried state must keep fixed shape/dtype. Sources:
  [JAX vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) and
  [JAX scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html).
- EnvPool/Brax/gymnax support the broader architecture lesson: environment
  throughput improves when env execution is deliberately parallelized and, for
  JAX systems, when environment and policy run together on the accelerator.
  Sources: [EnvPool arXiv](https://arxiv.org/abs/2206.10558),
  [Brax arXiv](https://arxiv.org/abs/2106.13281),
  [Brax GitHub](https://github.com/google/brax), and
  [gymnax GitHub](https://github.com/RobertTLange/gymnax).
- Mctx is relevant for search, not ray casting. It is JAX-native, JIT-friendly,
  and batched, so it is a good downstream consumer once `[B,2,106]` rows and
  masks are produced honestly. Source: [Mctx GitHub](https://github.com/google-deepmind/mctx).

## Next Experiments

- Re-run the source-backed circle-ray profile with current labels, body/trail
  counters, and observation phase timers.
- Add a dense batched ray-circle microbench that compares scalar parity,
  `[B,P,R,N]` memory use, p50/p95/p99 observation latency, and throughput for
  current `B`, `R=24`, and real body/trail counts.
- Add a Numba rung for the same inputs only if dense NumPy remains ray-bound or
  allocates too much.
- Add a simple grid/spatial-hash scout only if real body counts make dense
  all-body intersections worse than policy/search timing.
- Calibrate real model/search/Mctx timing on the same `[B,2,106]` rows before
  deepening CPU ray work.
- Do not start a GPU/JAX env rewrite until a report shows CPU env/obs plus
  transfers dominate a production-like loop after policy/search calibration.

## Fidelity Warnings

- Ray speedups are not source-fidelity proof. The scalar/source observer remains
  the parity oracle until Environment/RAM signs off on body/trail/bonus geometry.
- Broad phase must only prune candidates. Exact ray-circle intersection and
  channel semantics still need to match the trainer observation contract.
- JAX/GPU rewrites require fixed-shape tensor state, reset/autoreset/final
  observation semantics, masks, rewards, replay payloads, and parity tests.
- Host-device transfer can erase a fast GPU ray kernel if CPU env stepping,
  observation packing, policy/search, or action selection bounce per step.
- Bigger `B` is not automatically better on CPU; keep latency and policy
  freshness next to throughput.
