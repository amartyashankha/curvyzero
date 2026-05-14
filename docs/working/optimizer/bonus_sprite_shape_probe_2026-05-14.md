# Bonus Sprite Shape Probe - 2026-05-14

Scope: local inspection only. No production code edits, no Modal/live runs.

## Inputs

- Sprite sheet: `third_party/curvytron-reference/web/images/bonus.png`
- Current renderer mapping: `src/curvyzero/env/vector_visual_observation.py`
- Atlas layout used by renderer: `3 x 4`, row-major, `12` tiles, `100 x 100` px each.
- Optional visual artifact: `artifacts/local/curvytron_render_profiles/bonus_sprite_grid_enlarged_2026-05-14.png`

## Probe Method

Loaded the RGBA atlas with PIL/numpy. For each tile I measured:

- alpha support bbox and nonzero alpha pixels;
- effective alpha mass, `sum(alpha / 255)`;
- unique foreground RGB colors;
- coarse alpha signatures at `8 x 8` and `4 x 4`;
- coarse luma signatures after compositing on renderer background `(34,34,34)` and applying renderer luma weights `0.299r + 0.587g + 0.114b`.

This is an atlas-level signal probe, not a full in-game render parity test.

## Summary

- The reference contains `12` mapped bonus sprites.
- Every tile has bbox `(1, 1, 98, 98)`, so each sprite uses almost the full tile extent.
- Alpha support ranges from `1,997` to `2,456` px (`20.0%` to `24.6%` of a tile).
- Effective alpha mass ranges from `1,574.3` to `1,992.5` px-equivalent (`15.7%` to `19.9%`).
- Each tile has exactly one foreground RGB color: green `(44,203,58)`, red `(243,65,65)`, or gray `(204,204,204)`. Internal marks are carried by alpha/shape, not by a secondary color.
- Alpha-only coarse signatures are `11/12` unique at both `8 x 8` and `4 x 4`; `BonusSelfSlow` and `BonusEnemySlow` have identical alpha shapes and differ only by color/luma.
- Luma coarse signatures are `12/12` unique at both `8 x 8` and `4 x 4` in this isolated tile probe. The closest `4 x 4` luma pair is weakly separated: mean absolute difference `0.625` between `BonusEnemyFast` and `BonusEnemyInverse`.

## Per Tile

`a8/a4` are nonzero cells in coarse alpha signatures. `l8/l4` are cells whose luma differs from background by at least 2.

| idx | code | name | bbox | alpha px | alpha mass | rgb | a8 | a4 | l8 | l4 |
|---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|
| 0 | 3 | BonusSelfFast | `(1,1,98,98)` | 2365 | 1842.0 | green | 52 | 16 | 50 | 12 |
| 1 | 6 | BonusEnemyFast | `(1,1,98,98)` | 2360 | 1842.4 | red | 52 | 16 | 50 | 12 |
| 2 | 2 | BonusSelfSlow | `(1,1,98,98)` | 2280 | 1803.1 | green | 48 | 16 | 46 | 12 |
| 3 | 5 | BonusEnemySlow | `(1,1,98,98)` | 2280 | 1803.1 | red | 48 | 16 | 46 | 12 |
| 4 | 10 | BonusGameBorderless | `(1,1,98,98)` | 2096 | 1668.1 | gray | 48 | 16 | 48 | 12 |
| 5 | 4 | BonusSelfMaster | `(1,1,98,98)` | 2456 | 1946.6 | green | 52 | 16 | 50 | 12 |
| 6 | 7 | BonusEnemyBig | `(1,1,98,98)` | 2350 | 1992.5 | red | 52 | 16 | 50 | 12 |
| 7 | 11 | BonusAllColor | `(1,1,98,98)` | 1997 | 1574.3 | gray | 47 | 16 | 47 | 12 |
| 8 | 8 | BonusEnemyInverse | `(1,1,98,98)` | 2282 | 1776.3 | red | 51 | 16 | 48 | 12 |
| 9 | 1 | BonusSelfSmall | `(1,1,98,98)` | 2348 | 1892.5 | green | 52 | 16 | 50 | 12 |
| 10 | 12 | BonusGameClear | `(1,1,98,98)` | 2351 | 1900.8 | gray | 51 | 16 | 51 | 12 |
| 11 | 9 | BonusEnemyStraightAngle | `(1,1,98,98)` | 2159 | 1717.9 | red | 48 | 16 | 46 | 12 |

## Machine-Relevant Takeaway

The original sprites encode type information through both color/luma group and internal alpha pattern. A renderer that collapses bonuses to same-size filled circles discards the internal alpha pattern. A grayscale/downsampled path can still retain all 12 isolated tile signatures at coarse `8 x 8` and `4 x 4` levels here, but some pairs are very close at `4 x 4`, and alpha-only signal cannot distinguish the slow self/enemy pair.
