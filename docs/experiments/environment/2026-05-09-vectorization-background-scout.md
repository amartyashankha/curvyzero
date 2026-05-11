# 2026-05-09 Vectorization Background Scout

## Question

How should CurvyZero stay ready for NumPy, compiled CPU, JAX, PyTorch, or GPU
environment backends while the main lane finishes CurvyTron fidelity first?

This is planning only. It does not claim source-fidelity performance and does
not change environment fidelity code.

## Boundary

- Fidelity still leads. The current frontier is trail cadence first, then
  trail gaps/body absence, then crossing and collision safety.
- `curvyzero-v0` benchmark numbers are useful for benchmark scaffolding only.
  They are not CurvyTron source-fidelity numbers.
- Do not build `reset_many`, `step_many`, JAX, PyTorch, or GPU backends until
  single-env semantics and observation/reward contracts are pinned.
- Keep source runners, scenarios, common traces, and diff tools as evidence
  machinery. Trainers should still import only `curvyzero.env`.

## Current Code Shape

The current trainer-facing toy env is already partly array-shaped:

- `positions`: `(players, 2)` float array.
- `headings`: `(players,)` float array.
- `alive`: `(players,)` bool array.
- `death_tick`: `(players,)` int array.
- `occupancy`: `(height, width)` int grid.

The current hot path is still single-env Python:

- `step()` builds Python dicts for observations, rewards, terminal flags,
  truncation flags, and infos.
- `agents` rebuilds player ids repeatedly.
- `_physics_tick()` loops over players, actions, collision checks, and trail
  drawing.
- `_mark_segment()` samples each segment with `np.linspace` and writes cells one
  by one.
- `_observations()` builds a small flat vector and copies it once per agent.

Small local toy-v0 probe:

```sh
PYTHONPATH=src python3 scripts/benchmark_env.py --episodes 100 --max-steps 500 --format json
PYTHONPATH=src python3 -m cProfile -s cumulative scripts/benchmark_env.py --episodes 200 --max-steps 500 --format plain
```

Result: the 100 episode toy-v0 smoke ran `2,370` steps at about `35,146`
steps/s. The profile again points at `CurvyTronEnv.step`, `_physics_tick`,
`_draw_segments`, `_mark_segment`, `np.linspace`, `_mark_cell`,
`_observations`, and repeated `agents` construction. Treat this as a local
smoke signal only.

## Trail Semantics To Preserve

Trail cadence is the next fidelity slice, so future backend shapes should
already leave room for these source rules:

- Movement runs from elapsed milliseconds, speed, and heading.
- Server update order is reverse player order.
- Normal trail point insertion happens during `Avatar.update()`, after movement
  and before border/body collision.
- A normal point is inserted when `printing` is true and either there is no last
  trail point or distance from the last trail point is strictly greater than the
  avatar radius.
- Each emitted point synchronously materializes a world body when the game has
  started and the world is active.
- The live avatar body carries the current `bodyNum`; inserted bodies carry
  their own `num`.
- Own-body collision is delayed by point number:
  `currentBody.num - storedBody.num > trailLatency`.
- Opponent bodies can collide immediately.
- Circle collision is strict overlap. Exact tangent is safe.
- Print-manager gap toggles happen after collision and only if the avatar is
  still alive.
- Turning printing off emits an important boundary point, clears visual trail
  cursor/state, and leaves already materialized world bodies in place.
- While `printing` is false, normal per-radius point bodies are not added.
  Crossing empty hole space should be safe only when no old body is there.

The core lesson for vectorization: visual trail state and collision body state
are not the same thing. A future backend should model both explicitly.

## Data To Represent Later

Use fixed shapes per run profile. A good future state layout is structure of
arrays with a leading batch axis:

- `pos[B, P, 2]`, `heading[B, P]`, `alive[B, P]`.
- `score[B, P]`, `round_score[B, P]`, `death_tick[B, P]`.
- `move[B, P]` using source move ids internally, with public `0/1/2` action ids
  converted at the boundary.
- `printing[B, P]`.
- `last_trail_pos[B, P, 2]` plus `has_last_trail_pos[B, P]`.
- `trail_point_count[B, P]`.
- `body_num[B, P]` for the live body number used in collision checks.
- `body_count[B, P]` for the next inserted body number.
- `print_manager_active[B, P]`, `print_manager_distance[B, P]`,
  `print_manager_last_pos[B, P, 2]`.
- Explicit per-env RNG state for print/hole lengths, spawn variation, and
  future bonuses.
- Collision bodies as either a fixed body buffer or a grid/index:
  `body_pos[B, N, 2]`, `body_radius[B, N]`, `body_owner[B, N]`,
  `body_num[B, N]`, `body_active[B, N]`.
- Optional spatial grid or island index for CPU speed:
  fixed cells with body ids and masks, or an occupancy grid for simplified
  rulesets. Source-style circle bodies need owner, number, radius, and active
  masks, not just filled pixels.

For GPU/JAX friendliness, avoid variable Python lists in the hot state. Use
masks, counters, and fixed maximum body capacity per profile. Overflow policy
must be explicit and tested before it becomes a training backend.

## Operations To Batch

Batch these first in NumPy or compiled CPU form, before considering GPU:

- Convert actions to source moves for all live players.
- Update heading and position for all `B, P`.
- Compute normal trail insertion masks from `printing`, last trail position,
  radius, and moved position.
- Append materialized bodies for inserted normal points with body owner, num,
  position, and radius.
- Run border checks.
- Run body collision checks with strict circle overlap and own-body latency.
- Apply deaths, death-frame point materialization, and scoring/order rules.
- Run print-manager distance update and toggles only for survivors.
- Generate observations in batch, then let wrappers turn arrays into dicts only
  at the public edge.
- Reset/autoreset rows with masks instead of changing batch shape.

The hard part is same-tick ordering. A source-faithful backend may need a small
loop over players in reverse order inside each batch tick, while still batching
over environments. That is acceptable: batch over `B` first, keep `P` order
explicit, and measure before trying to remove that loop.

## What Can Stay CPU For Now

- JS oracle, scenarios, common-trace diffs, timelines, and source runners.
- Public single-env dict API and debugging adapters.
- Modal, artifact storage, batch summaries, and replay/debug output.
- Python reference runner for fidelity investigation.
- Wrapper output construction until measurements show it is a real bottleneck.

GPU stepping is not the next move. The branch-heavy collision and ordering rules
may run better as vectorized NumPy or compiled CPU kernels for a long time,
especially while model/search work also wants the device.

## Measurements Still Missing

The next benchmark data should be source-aware enough to guide backend choices:

- Movement-only time.
- Normal trail point insertion time.
- Body materialization and body buffer/grid update time.
- Circle collision lookup time, split by opponent bodies and own-body latency.
- Print-manager distance/toggle time.
- Observation generation time.
- Dict/wrapper output time.
- Reset/autoreset time.
- Memory per env for bodies, grid/index, observations, and scratch buffers.
- Scenario-mix throughput for scripted source fixtures before random-action
  smoke.
- Compile/setup/warmup time for any Numba/JAX/PyTorch experiment.

## Next Two Vectorization Tasks

1. Add a source-aware timing scaffold or isolated microbenchmark for the trail
   cadence slice. It should time movement, `isTimeToDraw`, point insertion,
   body materialization, and collision lookup separately, using scripted shapes
   that mirror the cadence fixtures.
2. Draft the fixed-shape batch state schema for trails and bodies. Keep it as a
   doc or small typed container sketch first: body buffer fields, masks,
   counters, own-latency fields, print-manager fields, and explicit overflow
   policy.
