# Bonus Symbol Render Plan

Date: 2026-05-14
Updated: 2026-05-15

Purpose: keep `simple_symbols` aligned with the current policy-observation
target.

## Current Target

`simple_symbols` is the bonus representation for the current policy-observation
target:

```text
GPU browser_lines + simple_symbols
```

CPU `browser_lines + simple_symbols` is the production backend and parity oracle
today. A future batched GPU backend must match it before replacing it.

`browser_sprites` are for artifacts, GIF/eval/reference views, and browser
fidelity checks. `body_circles_fast + simple_symbols` and `fast_gray64_direct`
are historical/prototype renderer names, not current training or tournament
policy surfaces.

Do not rely on `auto` defaults to express the target. Record resolved trail and
bonus modes explicitly.

## Implementation Facts

`simple_symbols` has two important shapes:

- the raw symbol mask is a `7x7` type code;
- the current CPU oracle draws that mask into the source-canvas-sized render
  and then the whole frame is downsampled to gray64.

Known code points:

- `BONUS_RENDER_MODE_SIMPLE_SYMBOLS = "simple_symbols"`;
- `BONUS_SYMBOL_OUTER_LUMA_BY_SHAPE = (68, 148, 196)`;
- `BONUS_SYMBOL_INNER_LUMA_BY_SHAPE = (212, 48, 48)`;
- minimum raw stamp footprint is `7x7` before scaling to the current render
  surface;
- marks are row-specific, so circle, diamond, and square rows do not reuse the
  exact same X/horizontal/vertical geometry.

Debug artifact:

```text
artifacts/local/curvytron_render_profiles/bonus_simple_symbols_actual_v8_20260514.png
```

The first implementation used one inner luma for every outer shape. Visual
inspection showed the square-row marks were too close to the square fill, so V8
uses per-shape inner luma and row-specific mark placement.

Relevant critiques:

- [actual visual critique](bonus_symbol_actual_visual_critique_2026-05-14.md)
- [actual numeric separability](bonus_symbol_actual_separability_2026-05-14.md)

Plain result: the raw/simple fast-path probes had no class collisions in the
tested center, offset, radius, edge-clipping, and 2P remap sweeps. The V8 raw
7x7 stamp nearest-pair margin was L1 `1300`; the mismatch-pixel floor was `10`.

## Symbol Code

The code is deliberately artificial:

```text
3 outer shapes x 4 inner marks = 12 classes
outer shapes: circle, diamond, square
inner marks: plus, cross, horizontal bar, vertical bar
```

Shape and luma both carry information. The luma values avoid body/head/remap
values that the player-perspective path rewrites.

Current code-order mapping:

| code | bonus type | symbol |
| ---: | --- | --- |
| 1 | `BonusSelfSmall` | circle + plus |
| 2 | `BonusSelfSlow` | circle + cross |
| 3 | `BonusSelfFast` | circle + horizontal bar |
| 4 | `BonusSelfMaster` | circle + vertical bar |
| 5 | `BonusEnemySlow` | diamond + plus |
| 6 | `BonusEnemyFast` | diamond + cross |
| 7 | `BonusEnemyBig` | diamond + horizontal bar |
| 8 | `BonusEnemyInverse` | diamond + vertical bar |
| 9 | `BonusEnemyStraightAngle` | square + plus |
| 10 | `BonusGameBorderless` | square + cross |
| 11 | `BonusAllColor` | square + horizontal bar |
| 12 | `BonusGameClear` | square + vertical bar |

## Drawing Rule

Render trails first, then bonuses, then live heads.

`simple_symbols` should overwrite every non-transparent stamp pixel on top of
the trail. Do not alpha-blend the readable symbol body by default: blending
would make the same bonus depend on the trail underneath it. If softer edges
are ever wanted, test only a thin edge treatment.

## Historical Measurements

These CPU measurements are useful speed history, not the current target:

- after the simple-symbol and minimum-footprint patch, local no-death env-only
  `body_circles_fast` measured about `1510` steps/sec for 500 steps and `1464`
  steps/sec for 1000 steps;
- matched `browser_lines` rows were about `351` steps/sec at 500 steps and
  `378` steps/sec at 1000 steps;
- env-only long trajectories showed about `3x` to `8x` speedups from the old
  fast gray64 approximation;
- the waited stock LightZero C8/sim8/no-death full-loop profile improved
  `36.55s -> 31.36s`, about `1.17x`, because search, policy forward,
  subprocess collection, replay, and learner work were still present.

Plain read: symbols are cheap enough to justify as the bonus target, but the
policy-observation destination is GPU `browser_lines + simple_symbols`, not
CPU `body_circles_fast`.

## Tests And Gates

Recorded checks:

- all 12 types rendered across offsets and radii;
- raw 7x7 base-symbol uniqueness pinned at minimum L1 `>=1300` and mismatch
  pixels `>=10`;
- pairwise direct-fast gray64 patches had no duplicate signatures;
- luma collision against player heads, trails, and background was checked;
- symbol-over-trail behavior was checked;
- render/wrapper slice after V8 recorded `81 passed, 1 skipped`.

Remaining gates for current policy use:

- prove CPU `browser_lines + simple_symbols` on real source-state rows;
- prove GPU equality against that CPU oracle;
- add or keep explicit metadata for resolved trail and bonus modes;
- include head-over-bonus, clipping, offset, radius, and player-perspective
  remap cases;
- run a stock full-loop profile before claiming end-to-end speedup.

## Non-Targets

- `browser_lines + browser_sprites` is not the policy-observation target.
- `body_circles_fast + simple_symbols` is not the recommendation target.
- Original sprites on GPU are a separate artifact/reference project, not a
  blocker for the current policy renderer.
