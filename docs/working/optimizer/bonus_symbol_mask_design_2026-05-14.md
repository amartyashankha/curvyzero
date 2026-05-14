# Bonus Symbol Mask Design

Date: 2026-05-14

Scope: proposal only. No production renderer, training, or Modal changes.

## Goal

Make the 12 CurvyTron bonus classes machine-separable in the final `64x64`
grayscale observation. This is not a pixel-art or human-icon parity project.
The design should be boring, redundant, easy to stamp on CPU/GPU, and measured
by crop separability after the same downsample/remap path used by training.

## Renderer Contract

Use a fixed `7x7` output-grid stamp centered on the bonus cell. The stamp has:

- an opaque diamond footprint: `abs(row - 3) + abs(col - 3) <= 3`;
- a group base luma across the whole diamond;
- an opaque internal binary mark, drawn over the base with a shared mark luma;
- no font glyphs, antialiasing, alpha blending, sprite sampling, or 255 outline.

If a canvas-sized CPU path is needed, draw the same logical `7x7` stencil as
solid integer blocks before area downsample. The direct fast/GPU path should
stamp the `7x7` `uint8` kernel directly. The symbol contract is the post-gray64
kernel, not the browser atlas.

## Luma Alphabet

Proposed exact luma values:

| use | luma |
| --- | ---: |
| internal mark | 48 |
| self bonus base | 116 |
| enemy bonus base | 172 |
| game/all bonus base | 204 |

Do not use exact values already carrying other semantics in the current paths:
background `34`, invalid/body/remap values `80, 96, 128, 160, 192`, legacy
uniform bonus `208`, heads `224, 232, 240, 248`, or white `255`. Also avoid
default player-color luma values around `76`, `150`, and `217`.

The luma bands are only coarse redundancy. Class identity comes from
`base_luma + fixed mark mask`, not from a single center pixel.

## Mark Atoms

Coordinates are `(row, col)` in the `7x7` stamp, zero-based. All atoms are
inside the diamond footprint and are drawn at luma `48`.

| atom | coordinates |
| --- | --- |
| `N` | `(1,2) (1,3) (1,4) (2,2) (2,3) (2,4)` |
| `E` | `(2,4) (2,5) (3,4) (3,5) (4,4) (4,5)` |
| `S` | `(4,2) (4,3) (4,4) (5,2) (5,3) (5,4)` |
| `W` | `(2,1) (2,2) (3,1) (3,2) (4,1) (4,2)` |
| `C` | `(2,3) (3,2) (3,3) (3,4) (4,3)` |

## Type Map

These are arbitrary machine IDs, not semantic pictograms.

| code | bonus type | base luma | symbol name | mark mask |
| ---: | --- | ---: | --- | --- |
| 1 | `BonusSelfSmall` | 116 | `self_n` | `N` |
| 2 | `BonusSelfSlow` | 116 | `self_e` | `E` |
| 3 | `BonusSelfFast` | 116 | `self_s` | `S` |
| 4 | `BonusSelfMaster` | 116 | `self_w` | `W` |
| 5 | `BonusEnemySlow` | 172 | `enemy_ne` | `N + E` |
| 6 | `BonusEnemyFast` | 172 | `enemy_es` | `E + S` |
| 7 | `BonusEnemyBig` | 172 | `enemy_sw` | `S + W` |
| 8 | `BonusEnemyInverse` | 172 | `enemy_wn` | `W + N` |
| 9 | `BonusEnemyStraightAngle` | 172 | `enemy_ns` | `N + S` |
| 10 | `BonusGameBorderless` | 204 | `game_ew` | `E + W` |
| 11 | `BonusAllColor` | 204 | `game_c` | `C` |
| 12 | `BonusGameClear` | 204 | `game_cns` | `C + N + S` |

This deliberately gives the five enemy classes larger marks than the four self
classes; enemy has the most classes in one luma band and needs more shape
entropy. `BonusGameClear` no longer uses `232`, avoiding the direct-fast
player-head remap caveat.

## Required Tests

- Atlas invariant: generated mask atlas is `(12, 7, 7)`, all marked pixels are
  inside the diamond, luma set is exactly `{48, 116, 172, 204}`, and none of
  those values are remap/head/body/background values.
- Centered separability: render all 12 bonuses with default radius around the
  arena center; require unique full-frame and crop hashes, minimum crop L1
  `>= 200`, and at least `10` mismatched crop pixels for every pair.
- Phase sweep: repeat over all `11x11` source-pixel downsample phases, several
  sub-cell world offsets, and radii near the source default, e.g. `2.5, 3.0,
  3.5`. No exact collisions; record the worst pair and reject if it falls near
  the browser-sprite probe margin.
- Edge/clipping sweep: place bonuses near each arena edge and corner. Partial
  clipping may reduce margins, but no two still-visible classes should collide.
- Remap sweep: run the player-perspective path for 2P and 4P rows, all
  controlled players. Pin `BonusGameClear`; its unobscured symbol pixels must
  stay in `{48, 204}` and must not be rewritten to `96` or `128`.
- Overlap stress: draw trails under the bonus and live heads tangent/partially
  overlapping it. Full occlusion is allowed to destroy identity; partial
  occlusion should not create class collisions until most of the footprint is
  covered.
- CPU/GPU parity: for the direct path, require exact `uint8` equality. For a
  canvas/downsample implementation, require equality to the CPU reference or a
  documented `<= 1` gray-level tolerance with the same rounding rule.

Critical read: this is only worth using if the phase/overlap tests show margins
far above the current browser-sprite nearest-pair result. If the margins are
thin, prefer the existing semantic bonus64-style planes over packing more bonus
identity into one grayscale channel.
