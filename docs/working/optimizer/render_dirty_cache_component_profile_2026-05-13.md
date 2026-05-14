# Dirty Cache Component Profile

Date: 2026-05-13

Scope: local optimizer profiling of the trusted full-fidelity CurvyTron visual
surface: `browser_lines` trails, browser-sprite bonuses, RGB canvas, BT.601
luma, 11x11 downsample to 64x64.

## Moving Target Warning

Environment Reconstruction is still landing fidelity changes. Treat every
number here as valid only for the code and render semantics that produced it.
If a profile shifts suddenly, first suspect that the environment or renderer
changed underneath the optimizer.

Every render profile should record:

- git status or commit if available;
- `source_state_trail_render_mode`;
- `bonus_render_mode`;
- `natural_bonus_spawn`;
- `death_mode`;
- trajectory length and warmup length;
- whether LightZero/search/learner are included;
- dirty-cache hit/fallback counts and fallback reasons;
- dirty-cache component timing buckets.

## Plain Definitions

`natural_bonus_spawn` is the real CurvyTron bonus system, not reward shaping.
When it is on, the source env schedules bonus timers, samples a bonus type and
position, inserts active pickup objects, and later expires or applies them.

In the full-fidelity visual path those active pickups are drawn as browser-like
sprite atlas tiles on the RGB canvas before luma/downsample. That is the path we
need for trusted visual training. Turning natural bonuses off is only an
ablation for attribution.

## What Was Timed

The first component timing was not a full training loop. It was local
environment stepping with no death and no LightZero MCTS/search/learner. The
number measures observation/render work inside env stepping.

Tiny smoke profile, 20 no-death steps, current local CPU:

| setting | wall | render/observation | simple read |
| --- | ---: | ---: | --- |
| natural bonuses on | 0.196s | about 0.13s | about 6.5ms/step render/obs |
| natural bonuses off | 0.101s | about 0.05s | about 2.5ms/step render/obs |

Dirty-cache component read from that tiny run:

- natural bonuses on: `draw_bonuses_sec` was about `0.067s` over 20 steps;
- natural bonuses off: `draw_bonuses_sec` was effectively zero;
- no-bonus next buckets were `trail_layer_update_sec`,
  `dirty_downsample_sec`, `compose_trails_sec`, and `dirty_blocks_sec`.

Plain read: active bonus sprite drawing is a real suspect in the current dirty
cache path. This is not yet the whole training bottleneck, and it should be
confirmed on longer profiles after reconstruction changes settle.

## Bonus Dirty-Block Fix

Fresh 100/500-step profiles showed the real problem more clearly. With natural
bonus spawning on, stationary bonuses were making the dirty cache mark their
sprite boxes dirty every step. At 500 no-death steps that inflated
`dirty_blocks_total` from `6,068` without bonuses to `162,185` with bonuses,
and render time rose from about `1.26s` to about `4.43s`.

The production patch now treats the active bonus list as a visual snapshot:
slot, id, type, position, and radius. Bonus boxes are marked dirty only when
that visible snapshot changes. Current active bonuses still expand any dirty
block that intersects them, so a new trail under a stationary translucent sprite
redraws the whole sprite box in the correct order. Bonuses that are unchanged
and far from dirty blocks are skipped, avoiding repeated alpha blending over
clean pixels.

After the patch, the same local 500-step profile with natural bonuses on
reported:

- `dirty_blocks_total`: `162,185 -> 12,153`;
- render time: `4.43s -> 1.54s`;
- wall time: `9.39s -> 6.13s`;
- `draw_bonuses_sec`: `0.300s -> 0.070s`.

The no-bonus 500-step control stayed around `6,068` dirty blocks and `1.32s`
render time. So the fix removes the runaway stationary-bonus cost, but active
bonuses still add real work when their boxes overlap dirty trails/heads.

Parity tests added around the risky cases:

- stationary far-away sprite does not drift the cached RGB frame;
- stationary sprite touched by a new trail redraws correctly;
- same position/radius with changed bonus type still dirties and redraws.

## Current Hypotheses

1. The biggest easy bug was stationary bonuses dirtying too many blocks. That
   is now patched and measured.
2. The next CPU wins are dirty block downsample, trail layer update, and a more
   direct block-local renderer that avoids composing full RGB frame regions.
3. Naive GPU full redraw is useful research but not the best production shape
   unless state and output stay on GPU or the render is dirty/incremental.
4. The renderer must stay visually faithful enough for the policy; bonus
   ablations are for attribution, not final training recommendations.

## Next Experiments

- Rerun 100/500-step no-death local profiles with natural bonuses on and off,
  after each meaningful reconstruction change.
- Add or keep Modal-safe telemetry that reports the last dirty-cache stats in
  profile runs.
- Prototype a bonus-sprite dirty-block path on CPU before trying to replace the
  full renderer.
- Prototype exact block-local 11x11 tile rendering: background, trail masks,
  sprite alpha, heads, luma/downsample, no full 704 frame allocation.
- Continue isolated GPU sprite research with a parity gate: one active sprite
  of each type, sprite over trail, and head over sprite.
