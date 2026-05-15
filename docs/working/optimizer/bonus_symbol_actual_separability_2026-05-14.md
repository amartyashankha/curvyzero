# Bonus Symbol Actual Separability

Date: 2026-05-14

Scope: local numeric toy probe against the latest code state only. No
production code edits, no Modal jobs, and no live runs.

## Status

Superseded by the V8 row-specific masks. This note is still useful historical
evidence for the luma bands, remap safety, edge-clipping risk, and the warning
that the RGB canvas/downsample symbol route collides at tiny or clipped
footprints. Current implementation truth is:

```text
artifacts/local/curvytron_render_profiles/bonus_simple_symbols_actual_v8_20260514.png
```

The in-repo V8 guard is
`tests/test_vector_visual_observation.py::test_direct_fast_simple_bonus_symbol_base_masks_keep_margin`,
which pins raw `7x7` uniqueness, L1 `>=1300`, and mismatch pixels `>=10`.

## What Was Probed

Inspected the actual implemented `BONUS_RENDER_MODE_SIMPLE_SYMBOLS` path in
`src/curvyzero/env/vector_visual_observation.py`. Current symbols are the
implemented `3` outer shapes x `4` inner marks family:

- outer luma values: `68, 148, 196`;
- inner mark luma by outer-shape row: `212, 48, 48`;
- base stamp: `7x7`, scaled by nearest-neighbor sampling for larger rendered
  radii;
- direct fast path enforces `radius_px >= 3`, so small radii still get at least
  a `7x7` final-gray stamp.

The regenerated visual artifact exists at
`artifacts/local/curvytron_render_profiles/bonus_simple_symbols_actual_20260514.png`;
this note is based on numeric rendering from the code path, not visual
inspection of the PNG.

Probe command:

```bash
uv run python /private/tmp/bonus_symbol_actual_separability_probe.py
```

Focused existing tests:

```bash
uv run pytest tests/test_vector_visual_observation.py -q -k simple_bonus_symbols
```

Result: `2 passed, 43 deselected in 0.09s`.

## Sweep

For all 12 source-default bonus type codes, the probe computed pairwise full
frame L1, mismatch-pixel counts, exact pair collisions, and exact hash
collisions.

Cases covered:

- centered/sub-cell offsets: offsets around `(32, 32)` at
  `(0,0)`, `(-0.49,-0.49)`, `(-0.49,0.37)`, `(0.25,-0.41)`,
  `(0.49,0.49)`, `(0.51,-0.51)`;
- radii: `0.25, 1.0, 2.0, 3.0, 3.1, 4.0`;
- edge clipping: sides, corners, and near-edge positions at radii
  `1.0, 3.0, 4.0`;
- 2P player-perspective remap with both players present/alive, placed away from
  the bonus.

## Results

| path / bucket | pair-case collisions | hash collisions | min L1 | min mismatch px | nearest pair / case |
| --- | ---: | ---: | ---: | ---: | --- |
| direct fast, center/offset/radius | 0 | 0 | 1000 | 10 | `BonusEnemySlow` vs `BonusEnemyBig`, small/default-radius centered stamp |
| direct fast, edge clipping | 0 | 0 | 300 | 3 | `BonusEnemySlow` vs `BonusEnemyBig`, corner-clipped stamp |
| direct fast, 2P remap | 0 | 0 | 300 | 3 | same clipped-corner nearest pair |
| canvas gray64 sanity path | 42 | 42 | 0 | 0 | small/downsampled or heavily clipped stamps |

Direct fast detail:

- No exact collisions were found in `72` center/offset/radius perspective cases.
- No exact collisions were found in `66` edge-clipping perspective cases.
- No exact collisions were found in `138` 2P-remap perspective cases.
- Small radii through `3.0` had the same direct-fast minimum margin:
  L1 `1000`, `10` mismatch pixels. Radii `3.1` and `4.0` expanded the stamp and
  improved the centered minimum to L1 `2000`, `20` mismatch pixels.
- Edge clipping is the weak point: corners can leave only `11` visible
  non-background bonus pixels and reduce the nearest-pair margin to L1 `300`,
  `3` mismatch pixels.

Canvas/downsample sanity detail:

- This is not the direct fast player-perspective path, but the same
  `simple_symbols` mode is callable through `render_source_state_canvas_gray64`.
- At radius `0.25`, the RGB-canvas symbol can downsample to only `1..4`
  visible final pixels; exact collisions appeared in `30` pair-cases.
- At radius `1.0`, exact collisions appeared in `12` pair-cases, including
  clipped-corner `BonusSelfFast` vs `BonusSelfMaster` collisions.
- At radii `3.0` and `4.0`, the canvas sanity sweep had no exact collisions,
  but clipped corners were still thinner than direct fast: minimum L1 `218` /
  `6` mismatch pixels for radius `3.0`, and L1 `455` / `13` mismatch pixels
  for radius `4.0`.

## Read

The actual direct fast `simple_symbols` implementation is numerically
separable across this toy sweep, including the 2P player-perspective remap.
The remap did not rewrite symbol pixels because the implemented symbol lumas
avoid the exact body/head remap values.

The latest contrast change materially improves direct-fast numeric margins.
The nearest direct-fast pair is now `BonusEnemySlow` vs `BonusEnemyBig`, and
the limiting contrast is the enemy-row inner-vs-outer gap `148 - 48 = 100`.
The corner margin is much stronger in L1 than before, but still spans only
three surviving pixels under severe clipping.

Important caveat: the canvas/downsample `simple_symbols` route is not robust at
small radii or severe clipping. If `simple_symbols` is intended only for the
direct fast gray64/player-perspective renderer, the probe is encouraging. If it
is intended to be a general replacement render mode through the browser-like
RGB canvas path, small-radius/downsample collisions need either a contract
change or explicit unsupported-case labeling.
