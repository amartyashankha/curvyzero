# Rendering And Raycasting Profile Plan - 2026-06-23

Status: source-read doctrine. No jobs were launched from this note.

## Short Answer

No, we do not need raycasting as a core assumption.

We should profile and compare raycasting because it might be a cheap useful
feature surface, especially for PPO/Puffer and planners. But the default
architecture should not depend on it. If raycasting loses on learning quality,
topology awareness, or full-loop economics, we can drop it without losing the
main plan.

Raycasting and rendering are different observation families:

- **Raycasting** returns structured distances from the player along fixed angles.
  It is geometry sensing, not pixels.
- **Rendering** returns raster images or image-like tensors. It is visual
  observation, even when the pixels are source-state approximations rather than
  browser-canvas pixels.

The useful question is not "which is faster" in one number. The useful question
is:

```text
which observation is cheap enough, faithful enough, and useful enough for the
agent or planner we are running?
```

Default posture:

```text
raster/source-state visual path = current trusted CurvyZero learning surface
raycasting = optional diagnostic/PPO/planner feature lane
Flash raycast = useful external control, not a requirement
```

Keep four ledgers separate:

1. Flash raw env/raycast controls.
2. CurvyZero flat ray trainer observations.
3. CurvyZero source-state visual renderers.
4. CurvyZero whole-loop MuZero/search training rows.

## What Raycasting Means Here

Flash `raycast_v1` is a GPU-resident structured observation. It starts with 20
scalar fields, then adds 13 player-relative ray angles:

```text
-90, -60, -45, -30, -15, -7.5, 0, 7.5, 15, 30, 45, 60, 90
```

Each ray reports normalized distance to:

- wall
- own trail
- opponent trail
- opponent head

The recovered Flash ABI documents this in
`artifacts/modal_deploy_downloads/curvytron-flash-app-20260623-101758/accelerated/ABI.md`.
The implementation is the Triton-backed
`curvytron_raycast_v1_observation2_kernel` in the recovered
`accelerated/kernels.py`, called by `AcceleratedCurvytronEnv` when
`observation_mode="raycast_v1"`.

CurvyZero also has a ray observation path:

- `src/curvyzero/env/vector_trainer_observation.py`
- schema `curvyzero_egocentric_rays/v0`
- 24 rays at 15-degree increments
- channels: wall/out-of-bounds, own trail, opponent trail, opponent head
- extra scalar state packed into a flat trainer observation

But this CurvyZero ray path is a narrow 1v1/no-bonus trainer-contract bridge.
It is not the current stock LightZero source-state visual policy surface.

## What Rendering Means Here

CurvyZero's current trusted visual policy surface is source-state raster
rendering:

- source-state-backed, not browser-pixel exact
- player-perspective 4-frame grayscale stacks shaped `[4, 64, 64]`
- `browser_lines` trail rendering
- `simple_symbols` bonus rendering
- CPU oracle / exact dirty-cache path as the trusted trainer-safe backend

Important local files:

- `src/curvyzero/env/vector_visual_observation.py`
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- `scripts/benchmark_render_lane_microbench.py`
- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`

The "super fast rendering" lane exists, but it is profile/canary evidence, not
the default production trainer path:

- `body_circles_fast` exists as a historical/control renderer mode, but current
  source-state LightZero tests reject it for the policy surface.
- `direct_gray64` is a fast policy-space GPU economics/profile surface.
- `block_704_gray64` is a closer 704-style benchmark surface.
- persistent JAX GPU framebuffer/profile work keeps trail layers on device and
  updates them incrementally, but remains profile-gated.

Do not call these browser-pixel parity unless a specific browser-canvas parity
artifact proves it.

## Current Evidence

Fresh Flash H100 control rows:

| Row | Observation | Speed | Read |
| --- | --- | ---: | --- |
| Flash grid baseline | no observations | `161.55M env/s` | mechanics ceiling |
| Flash grid compact obs | compact observation | `145.30M env/s` | cheap structured obs control |
| Flash grid raycast | `raycast_v1` | `15.65M env/s` | raycast observation-cost control |

Critical read: Flash `raycast_v1` is about 10x slower than bare mechanics in
that raw benchmark, but still very fast in absolute terms. It is not a CurvyZero
MuZero speed row.

CurvyZero renderer docs say the same caution from the other direction: raw GPU
drawing can be fast, but full-loop time often moves to host/device handoff,
stack update, scalar materialization, replay, search, and learner boundaries.
So a renderer-only win is useful only if it reduces the end-to-end observation
or search-input wall.

## Do We Need Raycasting?

Critical answer: no, not as a universal default.

We need raycasting only if it wins a specific job. The jobs where it is most
plausible are:

- **PPO/Puffer baseline observation.** A fixed flat vector is the easiest shape
  for a Puffer/Ocean-style C environment and recurrent PPO. It fits static
  buffers, action masks, MinGRU-style recurrence, and cheap policy inference.
- **Scripted opponents and safety policies.** Rays are a clean way to implement
  wall/trail avoidance, emergency steering, and diagnostic opponents such as
  `raycast_safety` without rendering images.
- **Planner features.** Time-to-wall/body along feasible directions is directly
  useful for macro-action pruning, beam/MPC scoring, and collision canaries.
- **Monitoring.** Ray features can expose whether a policy is entering
  near-collision states, escaping corridors, or simply surviving in open space.

The jobs where raycasting is weak are just as important:

- **Long-term topology.** Fixed rays are local. They compress global trail
  layout, enclosure structure, gap routes, and delayed dead ends.
- **Opponent strategy.** Rays see current geometry better than intent. They do
  not by themselves represent cut threats, baiting, or future opponent
  commitments.
- **Bonuses and hidden timers.** Ray distances do not naturally encode print
  state, hole timing, bonus timers, borderless effects, or inverse controls
  unless extra scalars are added.
- **Visual/RND continuity.** Current RND is built around visual policy frames.
  Ray-based RND would be a different feature source and could miss useful
  topology novelty that appears in maps.
- **Cost assumptions.** Raycasting is not free. In the Flash raw H100 controls,
  `raycast_v1` is about 10x slower than compact observations. It is still fast,
  but the slowdown is real.

So the right status is: raycasting is a candidate observation and diagnostic
surface, not a proof of a better agent.

## PufferLib Comparison

Puffer changes the raycasting question.

PufferLib's native path wants static contiguous buffers. That makes raycasting
attractive because a ray observation is small, fixed, and flat:

```text
[scalars, ray distances] -> fixed OBS_SIZE -> recurrent PPO
```

This is much easier to wire than a source-state image stack:

```text
[4, 64, 64] gray stack -> larger OBS_SIZE or conv frontend -> more buffer/copy/model cost
```

But the same criticism still applies. A Puffer ray baseline might be fast and
still strategically weaker than a visual or hybrid policy. It could learn
near-wall avoidance quickly while failing delayed enclosure, gap, or corridor
ownership scenarios.

For Puffer, the sensible ladder is:

1. **Compact scalar baseline**
   Cheapest sanity check. Likely too weak for serious play.
2. **Ray baseline**
   Best first serious Puffer observation because it is fixed-buffer friendly and
   exposes collision geometry.
3. **Ray plus strategic scalars**
   Add printing/gap/bonus/timer/opponent-motion features before adding pixels.
4. **Hybrid map or local crop**
   Add only if ray policies are myopic in scenario tests.
5. **Visual stack**
   Use if it beats ray/hybrid after accounting for buffer, model, and training
   cost.

Puffer does not make raycasting necessary. It makes raycasting the cheapest
credible first representation for a recurrent PPO/self-play baseline.

## What To Compare

Compare on three axes.

### 1. Observation Cost

Goal: price the surfaces without pretending this is learning quality.

Rows:

- Flash no-observation mechanics control
- Flash compact observation control
- Flash `raycast_v1` observation control
- CurvyZero flat egocentric ray batch observation
- CurvyZero CPU source-state gray64 stack update
- CurvyZero GPU render prototype: `direct_gray64`
- CurvyZero GPU render prototype: `block_704_gray64`
- CurvyZero persistent framebuffer/profile path

Metrics:

- env steps/sec
- policy rows/sec
- observation rows/sec
- H2D/D2H bytes and time
- device render time
- CPU packing time
- stack update time
- parity/fidelity status
- fallback path used or not

### 2. Representation Quality

Goal: ask whether the observation lets the agent solve CurvyTron.

Raycasts are cheap and geometric, but they compress global trail topology.
Visual stacks are richer and closer to the existing policy surface, but cost
more and can carry visual nuisance. A strong strategy may need either:

- raycast plus strategic scalar/auxiliary targets
- raster plus recurrence
- a hybrid map/crop/ray representation

Signals:

- survival under opponent pressure
- death cause split
- near-wall recovery
- corridor escape
- opponent cut/avoidance behavior
- action entropy and collapse
- fixed-opponent transfer
- policy-only versus planner-assisted decisions on fixed states

### 3. Whole-Loop Effect

Goal: avoid lying to ourselves with a fast observation row.

CurvyZero speed claims still need same-work H100 full-loop rows. The whole-loop
profile must include:

- env step
- observation/root-state construction
- search/action selection
- replay append/sample
- learner train
- refresh/cache/owner maintenance
- checkpoint/eval/proof context where relevant

If a renderer or ray path only wins in isolation, it stays a component result.

## Initial Profile Matrix

Run this as a staged funnel.

### Local, Cheap

- `uv run pytest tests/test_vector_trainer_observation.py tests/test_vector_visual_observation.py tests/test_benchmark_render_lane_microbench.py tests/test_source_state_gpu_render_benchmark_cpu.py -q`
- `uv run python scripts/benchmark_render_lane_microbench.py --plan grid --batch-sizes 16,64 --player-counts 2 --trail-lengths 64,512,2048 --kinds full_stack_update,gray64_render_only,perspective_reuse_gray64 --format json`

Use this to catch stale assumptions and obvious renderer regressions.

### H100 Component Controls

Small, parallel, profile-only:

- Flash raw no-observation rerun
- Flash compact-observation rerun
- Flash `raycast_v1` rerun
- CurvyZero `source_state_gpu_render_benchmark` direct/profile cells
- CurvyZero batched observation boundary profile with CPU oracle and GPU
  candidate surfaces

These can be broad and short. They are component rows.

### H100 Whole-Loop Rows

Only after component rows are healthy:

- stock source-state visual LightZero profile
- same profile with any alternate observation backend explicitly labeled
- compact/profile lanes only if they pass their existing same-work gates

These should be fewer and more controlled. They are the only rows that can
change the CurvyZero speed belief.

## Decision Rules

Promising:

- observation/profile win survives parity checks
- no hidden CPU fallback
- no stale resident frame
- H2D/D2H does not dominate
- whole-loop row improves, not just render kernel time
- learning/eval signal is at least neutral against the trusted observation

Not promising:

- renderer-only speedup vanishes in full loop
- raycast policy loses corridor/topology behavior that visual policy had
- GPU path needs per-step readback of full frames/stacks
- component benchmark changes observation semantics without relabeling
- Flash raw row is used as a CurvyZero MuZero speed claim

## Captain Read

Raycasting is worth keeping in the toolbox, but the burden of proof is on it.
It is especially useful for Puffer/PPO baselines, scripted opponents,
safety/planner features, and cheap geometric signals. It is not obviously the
right final observation for long-term CurvyTron strategy.

Rendering is also worth keeping, because CurvyTron is a topology game and the
visual stack can expose global trail shape. The current trusted CurvyZero
surface remains source-state gray64 visual stacks. The fast GPU/persistent
rendering lane is promising, but it must earn promotion through parity,
freshness, and whole-loop evidence.

The most believable next move is not to pick one forever. Profile both as
separate observation families, then let learning/eval decide whether the best
agent wants rays, raster, or a hybrid representation.
