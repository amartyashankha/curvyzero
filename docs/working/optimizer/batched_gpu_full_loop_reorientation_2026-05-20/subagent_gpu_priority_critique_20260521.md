# Subagent GPU Priority Critique

Date: 2026-05-21

Scope: Amdahl/prioritization critique using local docs and code context only.
No code changes, live trainer changes, launch default changes, tournaments,
checkpoints, eval jobs, or active runs touched.

## Bottom Line

The current plan should stop treating "make the renderer faster" as the main
optimizer story. With the user's current C512 anchor:

```text
C512 real observation:  ~1246 steps/s
C512 zero observation:  ~1826 steps/s
same-loop zero/real:    ~1.47x
```

That means the maximum possible gain from removing the current renderer/
observation path inside this same stock-loop shape is about `1.47x`, or about
`+46.5%` throughput. In wall-share terms, the removable observation slice is
about `31.8%` of the current real row:

```text
renderer/observation upper-bound share ~= 1 - 1246 / 1826 ~= 31.8%
```

That is worth taking if it is cheap. It is not an order-of-magnitude route. If
the target is multi-x speedup, the remaining `68%` of the loop must become the
priority: manager topology, policy/search batching, scalar LightZero payloads,
host/device synchronization, normal-death live-root collapse, and RND cadence.

## Amdahl Read

The zero-observation row is the cleanest local upper bound because it keeps the
stock LightZero collection/search/replay/learner boundary active while removing
most pixel work. It is still not literally "renderer only" because stack,
manager, scalar timestep, policy, search, replay, and learner work remain.
Therefore, `1826 steps/s` should be read as the same-architecture ceiling for
"observation is free-ish," not as evidence that another renderer kernel can
unlock more.

| anchor | steps/s | read |
|---|---:|---|
| C512 real | `1246` | current real observation throughput |
| C512 zero | `1826` | same-loop observation-free-ish ceiling |
| zero / real | `1.47x` | absolute upper bound for observation cleanup in this shape |
| real / zero | `68.2%` | non-observation floor already dominates |
| implied observation wall | `31.8%` | useful but not enough for `5-10x` |

If the current plan spends another phase on renderer-only details, the best
credible outcome is roughly reaching the zero row. That caps the same-loop
speed at about `1826 steps/s`. A `5x` result over `1246` would be about
`6230 steps/s`; a `10x` result would be about `12460 steps/s`. Renderer work
alone leaves roughly another `3.4x-6.8x` unexplained after the zero ceiling.

The older local docs point the same way. Prior C512 notes used `1439.84` real
versus `1805.22` zero, a smaller `1.25x` gap. The exact current pair is more
favorable to renderer work (`1.47x`), but still bounded. The conclusion does
not change: renderer cleanup is a contributor, not the architecture.

## Visible Bottlenecks Besides Renderer

The local notes show several walls that survive zero observation or reappear
after direct rendering gets cheap:

- **One-process batched manager ceiling.** C768 did not scale cleanly in the
  current docs: real was flat versus C512 and one zero row was slower than
  real. That is a scheduling/topology warning, not a pixel-render warning.
- **Subprocess parallelism was real.** Early C64 rows showed subprocess
  CPU-oracle beating the one-process batched GPU manager (`883.03` versus
  `416.89 steps/s`). The batched path improved with width, but it still gave
  up worker parallelism.
- **Policy/search are no longer background.** Once observation gets cheaper,
  policy forward, MCTS/root batching, and collector synchronization become the
  main moving pieces. Sim count sweeps already show non-monotonic behavior.
- **Scalar LightZero boundary remains hot.** The profile manager still returns
  Python dicts of scalar env ids, `BaseEnvTimestep`-like rows, per-row `info`,
  reward/done coercions, and host NumPy observation arrays.
- **Host/device ping-pong remains architectural.** Current GPU render lanes
  compact CPU env state, copy to device, render, block, copy frames back to
  host, update a host float32 stack, then let Torch policy/search upload
  observations again through the stock model path.
- **Payload/stack shape is large.** The direct-render notes say device render
  became small in the B512 surface canary (`~0.014s`) while the surface/payload
  boundary stayed large; one row records about `67.1MB` of float32 payload.
- **Normal death changes the workload.** Normal-death rows collapse live root
  batch size and trigger partial/autoreset/final-observation paths. A no-death
  renderer benchmark cannot predict that wall.
- **RND is a separate cadence wall.** RND meter rows are around `10-12%` slower
  in C512-style profiles, and small canaries show RND train can exceed render
  time at tiny batch sizes. Do not mix RND cost into renderer claims.

## Priority Critique

The current optimizer plan is right to keep the batched GPU manager as a
profile lane, but it should demote renderer-only work from "main quest" to
"bounded cleanup with a stop condition."

Recommended priority order:

1. Treat `1826 steps/s` as the C512 same-loop ceiling until a matched repeat
   disproves it. Do not promise speedups above that from renderer work.
2. Prototype a topology that can beat the zero-observation ceiling before
   spending deep effort on another render kernel.
3. Keep renderer cleanup opportunistic: direct observation, active-prefix
   render width, row-major output, and payload slimming are good only when they
   reduce the full-loop gap or unblock correctness gates.
4. Measure policy/search/root batching and manager wait as first-class timers
   in every serious row.
5. Keep RND, normal death, and renderer/backend changes as separate axes until
   each has matched controls.

The sharpest stopping rule: if C512 real rows land within roughly `10-20%` of
zero observation, renderer-only work should stop as the primary lane. If they
stay `30%+` below zero, renderer/stack/payload work is still valid, but still
cannot explain a multi-x target by itself.

## Prototype Next

The next prototype with enough Amdahl room is the profile-only hybrid actor
plus central batched observation harness described in the local plan:

```text
N CPU actor subprocesses step compact CurvyTron state
-> parent gathers compact state/reward/done/generation
-> parent batches zero observation first
-> if zero passes, parent switches to direct_gray64 batched render
-> scalar LightZero-shaped timesteps only at the outer edge
```

Run zero observation first. The purpose is to prove topology before render:
can actor parallelism plus central batching beat the current one-process C512
zero ceiling (`~1826 steps/s`) by at least `20%`, or scale upward where C768
one-process does not? If zero cannot beat zero, real render will not save the
shape.

Required prototype measurements:

- actor step time, actor idle/wait, parent send/receive, and compact payload
  bytes;
- parent gather/merge time;
- observation time split into zero/direct render, pack, H2D, device render,
  D2H, stack;
- scalar timestep and `ready_obs` materialization time;
- policy/search root batch size once stock-loop integration is attempted;
- live physical row count, terminal/autoreset count, and final-observation
  timing for normal-death gates;
- total effective env steps/sec against one-process real and zero anchors.

Parallel side experiments are still useful, but subordinate:

- C512/C768 real-vs-zero repeats with sim2/sim4 to confirm the plateau and
  search interaction.
- Search/root batching sweeps at C256/C512/C768 with matched workload counts.
- RND no-RND versus meter cadence rows, reported separately.
- A direct-render full-loop A/B only after the vector boundary preserves the
  batch and records exact backend identity.

## Do Not Change In Live Trainer Yet

Do not promote any of these to live training or defaults yet:

- `direct_gray64` or renderer-backed GPU observation as a trainer default;
- scalar `policy_observation_backend=jax_gpu` as a shortcut;
- the batched GPU profile manager as a Coach/live launch recommendation;
- DI-engine registry/default wiring outside profile-only harnesses;
- tournament, checkpoint, eval, GIF, or metadata consumers;
- positive-weight RND reward plumbing or normalization coupled to renderer
  experiments;
- a full JAX/MCTX rewrite as a "speed patch" before the stock-compatible and
  hybrid profile lanes are measured.

Profile-only stock `train_muzero` canaries are fine when they are explicitly
marked, fail closed on backend identity, and write no live-run artifacts. The
live trainer should remain on the trusted path until the batched/hybrid lane has
matched controls for normal death, final observation, RND latest-frame
extraction, row/player order, no hidden fallback, and full-loop wall-clock wins.

## Decision

Use the C512 `1246 -> 1826` gap as the renderer budget: about one third of the
current wall, maximum `1.47x` same-loop upside. That is too large to ignore and
too small to be the main plan. The next serious work should prototype topology
that can beat the zero-observation ceiling, while keeping renderer changes
profile-only and guarded by full-loop A/Bs.
