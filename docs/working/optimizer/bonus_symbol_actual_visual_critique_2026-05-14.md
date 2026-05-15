# Bonus Symbol Actual Visual Critique

Date: 2026-05-14

Scope: critique of the latest implemented `BONUS_RENDER_MODE_SIMPLE_SYMBOLS`
path and regenerated visual artifact only. No production code, Modal jobs, or
live runs were changed by this note.

## Status

Superseded. This note critiqued the earlier non-row-specific symbol masks.
Current implementation truth is V8:

```text
artifacts/local/curvytron_render_profiles/bonus_simple_symbols_actual_v8_20260514.png
```

Use [bonus symbol render plan](bonus_symbol_render_plan_2026-05-14.md) for the
current decision. The still-relevant parts of this note are the luma values,
the `7x7` direct-final-gray64 footprint, and the warning that the RGB
canvas/downsample path is not the trusted symbol route.

## Inspected

- Code: `src/curvyzero/env/vector_visual_observation.py`
- Artifact: `artifacts/local/curvytron_render_profiles/bonus_simple_symbols_actual_20260514.png`

The current implementation uses a `7x7` base stamp, nearest-neighbor scaling,
and a minimum final stamp radius of `3` pixels (`7x7` bbox). Outer luma by shape
is `(68, 148, 196)`. Inner mark luma is now shape-dependent:
`(212, 48, 48)`, so the circle family uses bright marks and the diamond/square
families use dark marks. Zero pixels in the stamp remain transparent over the
background luma `34`.

The regenerated PNG is `2120x184`, 8-bit grayscale, with 12 enlarged stamps. Each
symbol body is `168x168`, i.e. the actual `7x7` mask enlarged by `24x`.

## Findings

Verdict: the revised masks are visually and numerically much safer than the
earlier single-inner-luma version. They are reasonable for the direct final
`64x64` simple-symbol path. The remaining weakness is footprint imbalance, not
same-family mark collapse.

| codes | outer | final support | mark pixels | mark contrast | read |
| --- | --- | ---: | ---: | ---: | --- |
| `1-4` | circle, luma `68` | `29` px | `9` or `15` | `144` bright | Strong contrast, smallest support. |
| `5-8` | diamond, luma `148` | `37` px | `9` or `15` | `100` dark | Current nearest-pair family, but still comfortably separated. |
| `9-12` | square, luma `196` | `49` px | `9` or `15` | `148` dark | The previous square mark collapse is fixed. |

- Distinguishability: all 12 masks are exact-unique in the regenerated artifact
  and in a centered direct final-`64x64` render at `map_size=88`,
  `bonus_radius=3`.
- Pairwise margins: nearest direct final crop pairs are now diamond-family pairs:
  `5` vs `7` and `5` vs `8` have L1 `1000` over `10` pixels, max delta `100`.
  Square-family weak pairs are no longer close: `9` vs `11` and `9` vs `12` are
  L1 `1480`, max delta `148`.
- Footprint: the bbox is consistently `7x7`, satisfying the minimum footprint
  goal. Filled support is still not roughly equal by area: circle `29`, diamond
  `37`, square `49` non-background pixels. Square support is about `1.69x` circle
  support, so group identity still carries a strong mass cue.
- Thickness: horizontal and vertical bars are thick enough (`3x5`). Plus and `X`
  are still one-pixel-stroke marks in the final tensor, but with the new contrast
  they are usable for direct `64x64` stamping. The `X` remains the most fragile
  mark under clipping or overlap.
- `64x64` safety: for the intended direct gray64 path, default two-player
  geometry (`map_size=88`, source bonus radius `3`) produces the minimum `7x7`
  final stamp. Symbol values `{48, 68, 148, 196, 212}` avoid the exact current
  body/head/remap values, so direct player-perspective remapping should not
  rewrite them.
- Collapse concerns: there is still no circle-in-circle mark. Shape/mark collapse
  is largely addressed by making diamond/square marks dark. The only remaining
  collapse-like concern is partial occlusion: a clipped or overwritten one-pixel
  `X` can still lose identity faster than the bar marks.

## Caveat

This artifact and the sanity render validate the direct final-`64x64` path. If
`simple_symbols` is also used through the RGB canvas-like `704 -> 64` area
downsample path, run a separate phase sweep; the enlarged PNG does not prove that
resampled path.

One code-state nit: `BONUS_SYMBOL_INNER_LUMA` now aliases only the circle-family
inner value (`212`). Consumers/tests that mean "all inner lumas" should use
`BONUS_SYMBOL_INNER_LUMA_BY_SHAPE`.

## Recommendation

Keep the revised implementation as the explicit simple-symbol approximation. Add
a crop-distance regression gate around the current nearest pairs, plus edge and
overlap checks. If equal footprint matters for learning behavior, switch to a
fixed support mask later; otherwise the current version is visually separable and
safe enough for direct final `64x64` use.
