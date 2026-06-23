# GPU Render Next Phase

Date: 2026-05-20

## Current Truth

Matched H100 B64 steps256 surface rows:

- CPU dirty-cache surface median: `0.237s`
- GPU renderer-backed surface median: `0.144s`
- GPU device render median: `0.123s`

Plain read: the renderer-backed surface is a real profile win over the current
optimized CPU visual stack, about `1.65x` on this row. It is not a full
training-loop speedup claim.

The direct `direct_gray64` probe now passed through the same renderer-backed
surface canary after the direct-symbol fix. This is deliberately an approximate
learned-observation surface: it renders directly in policy space instead of
making a dense source-resolution canvas and downsampling it. It must not be
reported as browser-pixel parity.

Same H100 B64 steps256 surface shape:

- `block_704_gray64` surface step median: `0.144s`
- `direct_gray64` surface step median: `0.0339s`
- `block_704_gray64` device render median: `0.123s`
- `direct_gray64` device render median: `0.00973s`

Separate semantic canary: H100 adversarial two-view `direct_gray64 +
simple_symbols` matched the CPU-direct oracle exactly. Local tests also check
that all 12 bonus symbols stay distinct, bonus symbols overwrite trails, and
heads overwrite bonus symbols.

Plain read: the dense source-resolution GPU renderer was the local wall. The
direct learned-observation surface is about `4.6x` faster at the surface canary
and about `12.6x` faster inside device render. This still is not a full
training-loop speedup claim.

The current canary is profile-only. It exercises the trainer-visible
renderer-backed surface shape, but it does not run stock LightZero collection,
search, replay, learner updates, checkpointing, eval, or tournaments.

The current RND evidence is also canary evidence. RND hooks and a small meter
smoke ran, but this is not positive-weight RND learning proof and not a full
training result. Positive RND reward is still blocked on intrinsic
normalization and cap decisions.

## Why GPU Render Is No Longer The Only Wall

2026-05-21 supersession: keep this doc for renderer-specific evidence, but do
not use it as the current optimizer priority. Later H100 collect/MCTS splits
showed that renderer-only work is not the active wall in the current profile
shape. The current main lane is the LightZero MCTS branch representation path:
root setup, CPU/list conversion, result handling, and per-root output fanout.

At B64 steps256, surface overhead outside render is small:

```text
surface step median 0.144s
renderer median     0.130s
device render       0.123s
```

So the next drastic speedup has to mostly reduce device render work, not shave
tiny scalarization, stack, pickle, or wrapper costs.

That was true for the dense `block_704_gray64` renderer. The corrected
`direct_gray64` surface changes the read:

```text
B512 direct surface step median 0.123s
direct device render median     0.014s
payload pickle outside surface  0.020s
payload size                    67.1MB
```

Plain read: direct rendering removed most of the device-render wall in the
profile-only surface canary. The next wall is the boundary around it:
state/stack packing, large float32 payloads, subprocess/collector shape, RND
work, and whether the real stock loop can preserve the batch at all.

## Questions To Answer

- Can the device renderer avoid drawing inactive or overwritten trail work more
  aggressively without losing learned-observation semantics?
- Can dynamic render width become more granular than power-of-two S32/S64/S128
  without adding host overhead or branchy kernels that erase the win?
- Can line/body data be compacted on device once per batch instead of rebuilt
  or copied in host-heavy form each step?
- Can simple-symbol and browser-line rendering be split so unchanged/static
  pieces are cached or fused more cheaply?
- Does the renderer become memory-bandwidth bound, launch bound, or arithmetic
  bound at B64/S512 and at wider full-loop collector shapes?
- What render cost remains after full reset, terminal, partial autoreset,
  both-seat, row-order, and RND latest-frame gates are turned on?

## Hypotheses

- Biggest near-term win: make render cost proportional to active visible trail
  work, not env body capacity or coarse power-of-two padding.
- Second win: reduce per-step device work by caching or reusing static/simple
  symbol layers when the board state permits it.
- Third win: keep compact state resident on device across adjacent profile
  steps, then only update changed rows. This may be harder to reconcile with
  current Python/LightZero boundaries.
- Risk: a clever kernel can win the renderer microbenchmark and still lose the
  full loop if it adds synchronization, host copies, or complicated reset
  handling.
- Risk: RND can change the hot path. Treat RND as a separate meter until
  normalization and positive-reward training are proven.

## Next Research Wave Agents

- **Kernel profiler agent:** inspect device-render timing by trail width,
  active-prefix distribution, batch size, and dtype. Report whether the wall
  looks bandwidth, launch, or arithmetic dominated.
- **Algorithm agent:** look for render algorithms that skip inactive trail
  tails, avoid redrawing stable pixels, or use tighter per-row active spans.
- **State residency agent:** map what compact state can stay on GPU across
  steps, what must return to host for LightZero, and where synchronization
  currently happens.
- **Correctness gate agent:** finish reset, terminal, partial autoreset,
  row/player order, both seats, missing/extra actions, final observation, and
  RND latest-frame tests before promotion.
- **Full-loop A/B agent:** design the first base-manager full-loop profile
  comparing CPU oracle against the future batched GPU facade after correctness
  gates pass.

## No-Live-Run Rule

Do not change live training runs for this phase.

Do not promote scalar `policy_observation_backend=jax_gpu`.

Do not treat the current profile-only canary as RND training or full training.

Do not make renderer-backed GPU observation a trainer, tournament, or
checkpoint default until metadata, loader, reset/final-observation, RND, and
full-loop A/B gates agree.

## Immediate Next Step

Keep work in profile and docs until the correctness gate is covered. The direct
surface probe removed most of the device-render wall in the local canary. Next:
update the Amdahl read from a real full-loop profile before deeper kernel work.
If the full loop still shows observation as the wall, carry the direct surface
through vector-facade/full-loop correctness gates. If search/collector/RND
dominates, optimize there instead.
