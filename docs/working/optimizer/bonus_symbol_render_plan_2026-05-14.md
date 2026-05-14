# Bonus Symbol Render Plan

Date: 2026-05-14

Purpose: decide whether replacing exact bonus sprites with simple unique
symbols is a sensible speed/GPU lane.

## Plain Recommendation

Yes, this makes sense as an explicit opt-in experiment.

Do not replace the CPU reference renderer. Keep `browser_sprites` as the
trusted reference/default. Add a deliberate `simple_symbols` bonus mode only
after it passes visibility and parity tests.

The reason it can work: the policy ultimately sees `[4,64,64]` grayscale, not
the original sprite art. Exact sprite beauty is less important than preserving
the facts the policy needs:

- which bonus type it is;
- whether it helps self, hurts enemy, or changes the whole game;
- where it is;
- roughly how large/catchable it is;
- whether it remains visible after grayscale and downsample.

## Inventory

There are 12 active bonus sprites. The atlas is:

```text
third_party/curvytron-reference/web/images/bonus.png
300x400 RGBA, 3 columns x 4 rows, 100x100 per tile
```

The current Python renderer mirrors the atlas order in
`src/curvyzero/env/vector_visual_observation.py`.

| atlas index | bonus type | visual family |
| ---: | --- | --- |
| 0 | `BonusSelfFast` | green diamond, fast/curve internal mark |
| 1 | `BonusEnemyFast` | red diamond, fast/curve internal mark |
| 2 | `BonusSelfSlow` | green diamond, block/bar internal mark |
| 3 | `BonusEnemySlow` | red diamond, bar internal mark |
| 4 | `BonusGameBorderless` | gray diamond, border/internal square mark |
| 5 | `BonusSelfMaster` | green diamond, block/glyph internal mark |
| 6 | `BonusEnemyBig` | red diamond, solid diamond internal mark |
| 7 | `BonusAllColor` | gray diamond, paint/brush-like internal mark |
| 8 | `BonusEnemyInverse` | red diamond, inverse/loop internal mark |
| 9 | `BonusSelfSmall` | green diamond, small block internal mark |
| 10 | `BonusGameClear` | gray diamond, solid square internal mark |
| 11 | `BonusEnemyStraightAngle` | red diamond, angled-turn internal mark |

All of them are roughly diamond-framed icons with one internal symbol.

## Proposed Symbols

Think of this as a learned 12-class visual code, not human signage. A conv net
does not need the marks to be pretty or self-explanatory. Digits, glyphs, or
arbitrary masks can work if the final `[4,64,64]` observation consistently
separates the 12 bonus classes and correlates them with the right effects.

Use fixed binary masks, not font rendering. Use both shape and luma because
that gives redundancy. Shape-only can alias at small size. Luma-only can collide
with trail/head values or perspective remapping.

## Current Fast Encoding Critique

The current fast bonus encoding is not pure random garbage, but it is only
semi-principled. It uses evenly spaced grayscale fills:

```text
78, 92, 106, 120, 134, 148, 162, 176, 190, 204, 218, 232
```

That gives isolated bonus types different values, so the idea is reasonable.
But it was not chosen from a measured separability search over offsets, sizes,
heads, trails, downsample phases, or player-perspective remapping.

The 2026-05-14 signature probe found:

- centered browser sprites produced `12/12` unique frames, but some grayscale
  pairs were very close;
- centered circles/direct fast also produced `12/12` unique frames;
- `BonusGameClear` is a real warning case because luma `232` can be remapped as
  a player-head value in the direct fast player-perspective path, bringing it
  close to neighboring bonus values.

Probe note:
[bonus symbol signature probe](bonus_symbol_signature_probe_2026-05-14.md).

Plain read: circles are useful as a speed lens, but the final fast renderer
should not rely on luma-only identity. A small arbitrary 12-symbol code is fine
if it passes separability tests in the actual training observation path.

Suggested code-order mapping:

| code | bonus type | symbol idea |
| ---: | --- | --- |
| 1 | `BonusSelfSmall` | center dot or small square |
| 2 | `BonusSelfSlow` | thick horizontal bar |
| 3 | `BonusSelfFast` | right chevron |
| 4 | `BonusSelfMaster` | plus |
| 5 | `BonusEnemySlow` | thick vertical bar |
| 6 | `BonusEnemyFast` | left chevron |
| 7 | `BonusEnemyBig` | filled block |
| 8 | `BonusEnemyInverse` | X |
| 9 | `BonusEnemyStraightAngle` | L corner |
| 10 | `BonusGameBorderless` | four corner brackets |
| 11 | `BonusAllColor` | checker or split quadrant |
| 12 | `BonusGameClear` | slash-through block or eraser |

Use coarse luma bands by group, then shape for type:

- self bonuses: bright band;
- enemy bonuses: darker or mid band;
- game/all bonuses: separate mid/bright band.

Exact luma values should be chosen by a signature test, not guessed.

## Amdahl Read

This is not a guaranteed giant full-loop speedup.

Current measurements say:

- env-only long trajectories can get about `3x` to `8x` faster from a fast
  gray64 approximation;
- stock LightZero full-loop profiles only improved about `1.3x` to `1.5x`,
  because search, policy forward, subprocess collection, and learner work are
  still present;
- if render is `75%` of a workload, making render free has a `4x` theoretical
  ceiling, and a `10x` faster render gives about `3.1x`;
- if render is `50%`, making render free has a `2x` ceiling.

So the symbol lane is worth pursuing because long-survival policies may become
render-heavy, and because it may make GPU rendering simpler. It should not be
sold as a standalone `10x` full-training fix.

## Tests Before Training Use

- Render all 12 types at several sub-cell offsets and radii.
- Compare pairwise cropped `gray64` patches; no duplicate or near-duplicate
  signatures.
- Check position and footprint against `browser_sprites`.
- Check luma collision against player heads, trails, and background.
- Save a 12-symbol debug grid in code order and atlas order.
- If GPU is used, require CPU-symbol and GPU-symbol frame equality or a very
  tight tolerance.
- Record render mode and bonus render mode in metadata so train/eval cannot
  silently mismatch.

## Design Rule

Do not optimize for human-recognizable icon art. Optimize for machine-visible
class identity in the actual observation tensor.

The cleanest design is a deliberately artificial code:

- one stable footprint for all bonus objects;
- a few coarse luma bands that do not equal player body/head source values;
- one internal binary mask per class;
- tests across offset, size, overlap, and player perspective.

If that looks less like the browser art but gives cleaner class separation, it
may be better for training than the original downsampled sprites.

## Smallest Safe Implementation

1. Add an explicit `BONUS_RENDER_MODE_SIMPLE_SYMBOLS` next to
   `browser_sprites` and `circles_fast`.
2. Implement a small in-code `uint8[12, N, N]` mask atlas.
3. Reuse the same world-to-stamp placement as browser sprites.
4. Keep defaults on `browser_sprites`.
5. Profile env-only first, then stock full-loop.

## Open

- A design subagent is proposing exact simple masks and luma bands.
- A value-critique subagent is checking whether this lane is worth doing now
  versus keeping effort on broader stock LightZero scaling.
- A sprite-shape probe is verifying the original sprite geometry from
  `bonus.png`.
- Need to decide whether `simple_symbols` should be drawn through RGB then
  downsampled, directly into gray64, or both with a parity test.
