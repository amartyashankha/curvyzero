# Bonus Symbol Mask Critique

Date: 2026-05-14

Scope: design critique only. No production renderer, training config, Modal, or
live-run changes.

## Context Read

The current trusted model-facing path is:

```text
source-state RGB 704x704 canvas-like frame
-> browser bonus sprite tiles by default
-> BT.601 luma
-> exact 11x11 area average
-> uint8[1,64,64]
-> downstream [4,64,64] stack
```

The renderer already has a fast approximation, `circles_fast`, which gives
bonus types type-coded grayscale RGB fills. The direct fast player-perspective
path also draws type-coded luma circles directly into `64x64`, then remaps exact
player body/head gray values. The signature probe found all 12 centered bonus
types unique, but also found thin browser-sprite margins and a concrete remap
footgun: `BonusGameClear` at luma `232` can be rewritten as a player-head value.

So the design target is not human icon quality. It is robust class identity in
the final tensor after the exact render/downsample/remap path.

## Verdict on 3 Outer Shapes x 4 Inner Marks

I would not use `circle/diamond/square x plus/cross/horizontal/vertical` as the
primary 12-class design.

It is attractive because it is simple and Cartesian, but the product structure
creates obvious near pairs:

- same inner mark, different outer shape;
- same outer shape, `horizontal` vs `vertical`;
- `plus` vs `horizontal` or `vertical`;
- `cross` vs either diagonal surviving poorly after phase shifts or clipping.

At the default bonus scale the final object is only about a `6x6` or `7x7`
patch. A one-pixel-thick diagonal or endpoint is a weak feature. A circle and a
diamond at this size differ by only a few perimeter pixels; a square either has
too much mass or must be inset until it is close to the other shapes. If the
implementation goes through 704x704 and 11x11 averaging, phase and rounding can
erase exactly the pixels that distinguish those outer shapes. If the
implementation stamps directly at `64x64`, the marks are crisper, but then CPU
and GPU must agree on the post-gray64 stencil contract.

The proposed product also does not line up with bonus semantics. CurvyTron has
4 self classes, 5 enemy classes, and 3 game/all/clear classes. A strict 3x4
outer-shape grouping either splits enemy bonuses awkwardly or uses arbitrary
groups. Arbitrary groups are fine for a learned code, but then there is no
reason to prefer visually familiar outer shapes over masks chosen for measured
pixel distance.

## Recommended Mask Contract

Use a fixed post-gray64 `7x7` stamp, not three different outer silhouettes.

Recommended invariant:

- The support footprint is always the same filled diamond:
  `abs(row - 3) + abs(col - 3) <= 3`.
- The diamond has a group base luma.
- The internal mark is drawn over the base with one shared dark luma.
- Class identity is `base_luma + mark_mask`, not a center pixel or outline.
- The direct/GPU implementation should stamp this exact `uint8[7,7]` kernel.
- If a canvas/downsample CPU path is needed, stamp source pixels so the final
  `64x64` result matches the same logical kernel; do not depend on font,
  antialiasing, or subpixel sprite sampling.

Recommended luma values:

| use | luma |
| --- | ---: |
| mark | `48` |
| self base | `116` |
| enemy base | `172` |
| game/all/clear base | `204` |

Avoid exact values already carrying current renderer semantics:
background `34`, stale/unknown/body/remap values `80, 96, 128, 160, 192`,
legacy uniform bonus `208`, head values `224, 232, 240, 248`, and white `255`.
Also avoid grayscale values near default player lumas around `76`, `150`, and
`217`. Exact equality is especially dangerous in the direct player-perspective
remap, because the remapper rewrites all pixels with those exact body/head
values, not only pixels known to be players.

## Concrete 12-Mask Atlas

Coordinates are `(row, col)` in a `7x7` stamp, zero-based. All coordinates are
inside the fixed diamond and are drawn at luma `48`.

```text
diamond support:
...#...
..###..
.#####.
#######
.#####.
..###..
...#...
```

Mark atoms:

| atom | coordinates |
| --- | --- |
| `N` | `(1,2) (1,3) (1,4) (2,2) (2,3) (2,4)` |
| `E` | `(2,4) (2,5) (3,4) (3,5) (4,4) (4,5)` |
| `S` | `(4,2) (4,3) (4,4) (5,2) (5,3) (5,4)` |
| `W` | `(2,1) (2,2) (3,1) (3,2) (4,1) (4,2)` |
| `C` | `(2,3) (3,2) (3,3) (3,4) (4,3)` |

Recommended mapping:

| code | bonus type | base | symbol | mark |
| ---: | --- | ---: | --- | --- |
| 1 | `BonusSelfSmall` | `116` | `self_n` | `N` |
| 2 | `BonusSelfSlow` | `116` | `self_e` | `E` |
| 3 | `BonusSelfFast` | `116` | `self_s` | `S` |
| 4 | `BonusSelfMaster` | `116` | `self_w` | `W` |
| 5 | `BonusEnemySlow` | `172` | `enemy_ne` | `N + E` |
| 6 | `BonusEnemyFast` | `172` | `enemy_es` | `E + S` |
| 7 | `BonusEnemyBig` | `172` | `enemy_sw` | `S + W` |
| 8 | `BonusEnemyInverse` | `172` | `enemy_wn` | `W + N` |
| 9 | `BonusEnemyStraightAngle` | `172` | `enemy_ns` | `N + S` |
| 10 | `BonusGameBorderless` | `204` | `game_ew` | `E + W` |
| 11 | `BonusAllColor` | `204` | `game_c` | `C` |
| 12 | `BonusGameClear` | `204` | `game_cns` | `C + N + S` |

This deliberately gives the five enemy classes more mark entropy than the four
self classes. It also keeps `BonusGameClear` away from luma `232`, the remap
caveat observed in the signature probe.

## If We Keep 3x4 Anyway

If the product-code approach survives for implementation simplicity, make these
changes before testing it:

- Use filled silhouettes, not outlines. Outlines are too fragile at `6x6/7x7`.
- Make the inner marks two pixels thick where possible.
- Do not use `plus`, `horizontal`, and `vertical` in the same outer-shape group
  unless luma also separates them; `plus` is just `horizontal + vertical`.
- Treat `cross` as risky unless the diagonal has at least six marked pixels
  after final downsample.
- Assign luma by semantic group even if outer shape is arbitrary.
- Require a measured minimum pairwise crop margin; do not accept visual
  distinctness by inspection.

For a forced 3x4 code, use outer shape only as a redundant bit, not the primary
class signal. The primary signal should still be the luma band plus an internal
mask with enough Hamming distance.

## Obvious Failure Cases to Test

- Downsample phase: one-pixel marks can disappear when a 704x704 stamp straddles
  different 11x11 blocks.
- Direct remap collision: any exact symbol value equal to body/head luma values
  can be rewritten during player-perspective remap.
- RGB palette reuse: a grayscale symbol RGB must not equal a player RGB, the
  background, or reserved fallback gray such as `(120,120,120)`.
- Edge clipping: outer-shape identity collapses first near walls and corners.
- Head occlusion: current draw order puts bonuses before live heads, so a
  catching player can erase the center of the symbol.
- Trail overlap: trails are drawn before bonuses in the canvas path, but dirty
  block and direct paths still need explicit overlap tests.
- Multi-bonus overlap: if two bonuses overlap, draw order or slot/id priority
  can replace pixels and create mixed-code patches.
- Radius variation: radii around the default, e.g. `2.5`, `3.0`, and `3.5`,
  should not collapse masks to identical crops.
- CPU/GPU parity: if GPU stamps final masks directly but CPU draws through RGB
  and downsample, the two paths need exact equality or a documented tolerance.

## Promotion Gate

Before training uses this, require:

- `12/12` unique crop hashes over centered, phase-swept, edge, and radius-swept
  placements.
- Pairwise crop L1 and mismatch-pixel margins comfortably above the browser
  sprite nearest-pair probe result, not just nonzero.
- A 2P and 4P player-perspective remap sweep, pinning `BonusGameClear`.
- Metadata naming the bonus render mode so train/eval cannot silently mismatch.
- A local speed result proving this beats existing `circles_fast` or clearly
  unblocks the broader GPU/compiled renderer.

Bottom line: a simple mask atlas is reasonable, but optimize the masks as a
machine-visible code. The 3x4 human-icon set is likely more fragile than a
fixed footprint with luma-banded, high-distance internal marks.
