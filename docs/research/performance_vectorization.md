# Simulator Performance And Vectorization

Source read: `curvytron_muzero_modal_handoff.md`, Version 2, May 8, 2026.
Updated with local toy-v0 scout and JAX/GPU source pass on May 9, 2026.

## Short Answer

Build the simulator so it can move from a clear Python reference path to a batched array hot path without changing game semantics. The first serious implementation should be Python/NumPy with fixed-shape state arrays, an occupancy-grid collision backend, deterministic golden tests, and a throughput benchmark. Keep the hot-loop interfaces Numba-friendly from the start, because Numba is the most likely first optimizer if NumPy plus small Python loops miss the target.

Do not start with a JAX-native environment, PyTorch tensor environment, or C++/Rust extension. Those may become useful, but only after measuring whether environment stepping, collision rasterization, observation generation, MCTS/model inference, or replay I/O is the actual bottleneck. MuZero search uses learned dynamics inside the tree, so the real simulator does not need to be JAX-native just because Mctx might be.

The 2026-05-09 local scout reinforces this: the current toy-v0 single-env smoke
is a useful regression signal, not a production performance target. A 1,000
episode local run reported 23,423 steps at 34,962.5 steps/sec, and cProfile
pointed first at segment/trail rasterization, then observation copies and wrapper
objects. Details: [toy-v0 performance scout](../experiments/environment/2026-05-09-toy-v0-performance-scout.md).

## Performance Goal

The v0 simulator should support thousands of simultaneous rollouts per host process or process group. That means the design has to minimize per-environment Python dispatch, keep memory bounded, batch observations/actions, and avoid network or storage calls inside the decision loop.

Initial useful measurements:

| Metric | Why it matters | First gate |
| --- | --- | --- |
| Physics ticks/sec/core | Measures the collision and movement floor before wrappers or learning. | At least 100k simple ticks/sec/core for v0 scenarios. |
| Vector env decisions/sec | Measures action-repeat plus observation overhead. | Enough to generate millions of decisions/day on one Modal job. |
| Observation generation time | Local rasters/rays may cost more than movement. | Report separately from physics. |
| Collision time share | Identifies whether occupancy updates dominate. | Profile wall, self, opponent, and head-head cases. |
| Memory per rollout | Caps feasible batch size. | Measured for grid, state, observations, and replay staging. |
| Compile time vs steady state | Matters for Numba/JAX experiments. | Report separately from runtime throughput. |

## Backend Comparison

| Backend | Strengths | Costs and risks | Best use |
| --- | --- | --- | --- |
| Python loops | Easiest to read, debug, and align with golden tests. Handles edge-case logic naturally. | Per-env and per-player dispatch will not scale to thousands of rollouts. Easy to accidentally put dicts/objects in the hot path. | Reference implementation, rule extraction, test oracle, debug renderer support. |
| NumPy | Good first batched CPU path. Fixed arrays can update positions, headings, masks, rewards, and many collision checks in bulk. Portable and easy to benchmark. | Scatter/raster updates for thick curved trails can reintroduce Python loops. Large occupancy grids can become memory-bandwidth bound. Variable episode lengths need masks/autoreset logic. | First production simulator path, especially with chunked vector envs. |
| Numba | Compiles explicit loops over envs, players, and rasterized trail cells. Can keep collision code simple while removing Python overhead. Good fit for CPU rollout workers. | Requires simple typed arrays and stable signatures. Compile latency and debugging are worse than Python. Some NumPy patterns and dynamic objects will not work. | First optimizer after the NumPy/reference path, if profiling shows Python or raster loops dominate. |
| JAX-native env | Pure `jit`/`vmap` stepping can run many envs with static shapes and may compose with JAX/Mctx tooling. | Occupancy-grid scatter updates, dynamic resets, control flow, and debugging are awkward. Shape changes trigger recompiles. GPU env stepping can compete with model/search work for memory bandwidth. | Later experiment if CPU env stepping is a measured bottleneck or if an all-JAX actor loop clearly simplifies the system. |
| PyTorch tensor env | Natural if the final trainer is PyTorch/LightZero and tensors are already batched. Can keep observations near the model. | Tensor control flow and scatter-heavy collision logic can be clumsy. CPU Torch may be slower than NumPy/Numba for simulator-style branching. GPU launch overhead can dominate tiny steps. | Later experiment only if the PyTorch path wins and data movement becomes a measured cost. |
| C++/Rust extension | Highest ceiling for CPU collision/raster loops, memory layout, bitsets, SIMD, and thread pools. Clean boundary can expose a fast `step_many`. | Packaging, CI, Modal builds, debugging, and cross-platform maintenance get heavier. Easy to prematurely freeze semantics before golden tests are mature. | Last-mile optimization after Python/NumPy/Numba profiles prove the simulator is the bottleneck. |

## Collision Model

Use an occupancy grid as the likely training hot path, but keep a geometry collision path as an oracle or benchmark target.

| Model | Strengths | Costs and risks | Recommendation |
| --- | --- | --- | --- |
| Occupancy grid | Constant-time lookup against existing trails and walls. Easy to vectorize or JIT. Natural for egocentric raster observations. Good for thousands of rollouts if grid size and dtype are controlled. | Approximation quality depends on resolution, trail thickness, and swept rasterization. Memory scales as `batch * height * width * channels`. Head-head and same-tick writes need explicit two-phase logic. | Default v0 hot path. Define exact grid units and validate with golden tests. |
| Continuous geometry | More faithful to curved/thick-trail semantics. Swept-circle or segment tests can catch tunneling and grazing precisely. Useful for explaining edge cases. | Naive segment checks are `O(history)`. Spatial indexes add complexity. Harder to batch across many envs and variable trail histories. | Build as a small reference/oracle only if needed for fidelity tests, not as the first hot path. |
| Hybrid | Occupancy grid for training speed, geometry checks for sampled validation and ambiguous cases. | Requires keeping two implementations semantically aligned. | Best long-term posture if fidelity matters. |

For the occupancy path, collision should be computed in phases:

1. Move all alive players using the same action tick.
2. Rasterize each swept head movement into candidate cells without mutating the grid.
3. Detect wall, old-trail, self-trail, opponent-trail, and head-head conflicts.
4. Resolve same-tick deaths and rank/scoring ties deterministically.
5. Write surviving or all trail cells according to the chosen rule.

The exact treatment of death trails, spawn gaps, wall inclusivity, and same-tick ties should be part of the environment config hash and golden tests.

## State Layout For Thousands Of Rollouts

The simulator hot state should be structure-of-arrays, not object-per-env:

```text
positions_x      float32[batch, players]
positions_y      float32[batch, players]
heading          float32[batch, players]
alive            bool[batch, players]
death_tick       int32[batch, players]
score            float32[batch, players]
rng_state        uint32/uint64 or backend key arrays with leading batch axis
grid             uint8 or uint16[batch, height, width]
tick             int32[batch]
done             bool[batch]
```

Design consequences:

- Use masks for dead players and finished episodes instead of removing rows.
- Prefer fixed maximum players per compiled/run configuration.
- Use `actions[batch, players]`, not per-agent dicts, inside the hot path.
- Carry body/trail counters as fixed arrays. Do not put variable Python trail
  lists in the future hot path.
- Use fixed occupancy grids and/or fixed trail/body buffers with active masks,
  owner ids, age/num counters, and write cursors.
- Keep Gymnasium, PettingZoo, and LightZero adapters outside the core step function.
- Preallocate scratch arrays for candidate cells, collision flags, observations, and rewards.
- Chunk large batches, for example 512 or 1024 envs at a time, when the occupancy grid or observations exceed cache/GPU memory.

Memory pressure is a first-class design constraint. A `4096 x 256 x 256` `uint8` grid is about 256 MiB before owner channels, observations, replay staging, or Python overhead. A `uint16` owner/time grid doubles that. This does not rule out large batches, but it argues for explicit chunking and measured grid sizes.

## Observation And Action-Repeat Constraints

Action repeat is part of the performance design, not only the game design. One model decision every 3 to 5 physics ticks can reduce MCTS and policy calls, but it changes control feel and collision frequency. Make it config-driven and benchmark multiple values.

Observation generation must be measured separately. Egocentric local rasters and ray features are promising, but local crops, rotations, and per-player perspective transforms can dominate runtime if implemented as per-agent Python work. The observation API should support batched generation:

```text
observe_many(state, perspective_players) -> obs[batch, players, ...]
```

Use smaller, stable observation shapes first. Avoid raw full-board image observations until a baseline proves the compact observation is insufficient.

## JAX And GPU Shape Guidance

This is a future-compatibility rule, not a near-term backend request.

JAX's `vmap` and automatic vectorization favor functions that operate on arrays
and can accept a leading batch axis. JAX `lax.scan` fits fixed-length rollout
loops where the carried state has the same structure, shape, and dtype every
iteration. JAX random APIs require explicit keys rather than hidden global RNG.
For CurvyZero, that means a future compiled transition should look like:

```text
step_arrays(state, action, rng_key) -> new_state, obs, reward, done, info_arrays
```

CurvyTron-specific consequences:

- Fixed max players per run profile, with masks for inactive players.
- Fixed map/grid size per run profile.
- Fixed-size occupancy grid and/or fixed trail/body buffers.
- Body/trail counters, print-gap state, score state, death state, and RNG state
  live in arrays.
- Random print holes, spawn variation, and domain randomization consume explicit
  per-env RNG state.
- Observation generation is batched and shape-stable.
- Same-tick collision and death-trail behavior stay two-phase and fixture-gated.

Mctx is still a separate search lane. Its MuZero/Gumbel MuZero APIs are
JAX-native, JIT-friendly, and batched, and the recurrent function consumes an
action plus learned embedding and returns reward, discount, prior, value, and a
new embedding. That does not force the real rollout simulator to be JAX-native
now. It does mean that fixed observation/action/reward shapes will reduce future
glue work.

EnvPool shows a different later option: CPU C++/threadpool batched env execution
can matter when parallel environment stepping is the bottleneck. Keep that as an
escape hatch after Python/NumPy/Numba evidence, before assuming a GPU simulator
is needed.

gymnax and Brax are useful patterns, not dependencies. gymnax shows explicit
state/params/RNG with `jit`, `vmap`, and `scan`. Brax shows accelerator simulation
is possible for some physics workloads, but its README warns that the old Brax
env side is not the path to copy for new environment work.

## Design Now

Build these constraints into the first simulator API:

| Area | Design now |
| --- | --- |
| Core API | `reset_many` and `step_many` over fixed-shape arrays, plus a clear single-env wrapper for tests. |
| Correctness path | A readable Python reference implementation and golden tests for wall, self, opponent, head-head, same-tick death, tunneling, and grazing. |
| Hot state | Structure-of-arrays state with no Python dicts, objects, or variable-length lists in the inner loop. |
| Collision interface | Backend boundary for `occupancy`, `geometry_oracle`, and future native/JIT backends. |
| Config/versioning | Hash grid size, coordinate units, trail thickness, action repeat, collision backend, scoring, tie policy, and observation schema. |
| Benchmark harness | Stable random-agent and scripted-action scenarios that report physics, collision, observation, wrapper, and memory metrics. |
| Determinism | Seed contract for spawn positions, reset batches, opponent sampling, and tie resolution. |
| Wrappers | Keep Gymnasium/PettingZoo/LightZero adapters outside the simulator core. |
| Modal shape | Keep stepping, model inference, and search local to a job/process group; no Modal Queue/Dict calls per tick or per MCTS node. |

## Postpone Until Measured

Do not commit early to these:

| Item | Defer until |
| --- | --- |
| JAX-native environment | CPU NumPy/Numba stepping is proven to bottleneck JAX/Mctx self-play. |
| PyTorch tensor environment | LightZero/PyTorch wins and host-device transfer is measured as costly. |
| C++/Rust extension | Numba or well-written NumPy cannot hit rollout targets and profiler points at collision/raster loops. |
| Continuous geometry hot path | Occupancy-grid fidelity fails important golden cases or training exploits grid artifacts. |
| Bitpacking/SIMD/spatial indexes | Grid memory or collision rasterization is a measured ceiling. |
| GPU-resident simulator | CPU rollout workers cannot keep the GPU search/trainer fed. |
| Distributed actor architecture | A single Modal job/process group has known bottlenecks and replay/checkpoint integrity is stable. |
| All-player MCTS | One searched ego plus policy/checkpoint opponents is too weak after batching works. |
| Bonuses/powerups | 1v1 no-bonus baseline and MuZero smoke are working and profiled. |

## Recommended Path

1. Implement a Python reference simulator and tests for semantics.
2. Implement a NumPy batched simulator using the same config and fixtures.
3. Add an occupancy-grid collision backend with two-phase same-tick collision resolution.
4. Add a benchmark script before optimizing, with separate timers for movement, collision, observation, wrappers, and reset/autoreset.
5. If NumPy misses the target, port only the hot collision/step kernel to Numba while preserving the same state arrays and tests.
6. Record backend shape assumptions in benchmark manifests: batch size, player
   count, map size, observation shape, action repeat, RNG shape, trail/body
   buffer shape, and config hash.
7. Run JAX/Mctx and PyTorch/LightZero spikes independently of simulator backend choice.
8. Consider JAX-native, PyTorch-native, EnvPool-like C++/threadpool, C++ or Rust
   only when profiler data says the environment is the limiting part of the
   training system.

## Open Questions

- What grid resolution and trail thickness best match the intended CurvyTron behavior while keeping memory reasonable?
- Should trails from players who die on a tick be written before the round terminates?
- Is the first learnable observation a local raster, ray features, or a hybrid?
- What action-repeat value balances control fidelity and MCTS cost?
- How many envs per process fit comfortably once observation and replay staging are included?
- Does vectorized observation generation or collision update dominate the first benchmark?

## Sources

- `curvytron_muzero_modal_handoff.md`
- `docs/research/training_architecture_notes.md`
- `docs/research/environment/performance_vectorization_plan.md`
- `docs/experiments/environment/2026-05-09-toy-v0-performance-scout.md`
- [JAX vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)
- [JAX automatic vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html)
- [JAX lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [jax.random](https://docs.jax.dev/en/latest/jax.random.html)
- [JAX pseudorandom numbers](https://docs.jax.dev/en/latest/random-numbers.html)
- [Mctx README](https://github.com/google-deepmind/mctx)
- [EnvPool NeurIPS 2022 abstract](https://papers.nips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html)
- [EnvPool docs](https://envpool.readthedocs.io/)
- [gymnax README](https://github.com/RobertTLange/gymnax)
- [Brax README](https://github.com/google/brax)
