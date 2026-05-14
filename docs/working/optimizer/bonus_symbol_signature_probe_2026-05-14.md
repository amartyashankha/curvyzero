# Bonus Symbol Signature Probe

Date: 2026-05-14

Scope: local, read-only/toy probe for whether the 12 source-default CurvyTron
bonus types remain distinguishable in gray64 under the current reference and
fast render paths. This does not change production code and does not replace the
CPU reference.

## Commands

Focused existing tests:

```bash
uv run pytest tests/test_vector_visual_observation.py -q -k "bonus_type_codes_map_to_browser_atlas_tiles or uses_sprite_bonus_path_by_default or renders_distinct_source_bonus_sprite_patches or bonus64_v1_distinguishes_all_source_default_map_bonus_types"
```

Result: `15 passed, 27 deselected in 0.15s`.

One-off inline Python probe:

```bash
uv run python - <<'PY'
# Synthesized one no-player/no-trail source-state row per bonus type.
# map_size=88.0, bonus_pos=(44.0, 44.0), bonus_radius=3.0.
# Rendered each row with browser_sprites gray64, circles_fast canvas gray64,
# and direct fast player-perspective luma64. Then hashed frames and computed
# pairwise L1/L2/mismatch-pixel distances.
PY
```

## Renderer Facts Inspected

- The current trusted gray64 path is
  `render_source_state_canvas_gray64`: RGB canvas-like 704x704, then BT.601
  luma and exact 11x11 area average to `[1,64,64]`.
- Default bonus mode is `browser_sprites`.
- `circles_fast` draws a white outline plus type-coded grayscale RGB fill before
  downsample.
- `render_source_state_gray64_fast_player_perspectives` draws type-coded luma
  circles directly at 64x64, then remaps player body/head gray values for the
  controlled-player perspective.
- Sprite atlas order in `bonus.png` is:
  `BonusSelfFast`, `BonusEnemyFast`, `BonusSelfSlow`, `BonusEnemySlow`,
  `BonusGameBorderless`, `BonusSelfMaster`, `BonusEnemyBig`, `BonusAllColor`,
  `BonusEnemyInverse`, `BonusSelfSmall`, `BonusGameClear`,
  `BonusEnemyStraightAngle`.

## Results

All three probed paths produced 12 unique full-frame hashes at the centered
source bonus position. Exact full-frame collisions were not observed.

| mode | unique hashes | object footprint | nearest full-frame pair |
| --- | ---: | --- | --- |
| `browser_sprites` -> gray64 | 12/12 | 24 non-background pixels, bbox 6x6 | `BonusSelfSmall` vs `BonusSelfSlow`: L1 22, 7 mismatch pixels, max abs 6 |
| `circles_fast` -> gray64 | 12/12 | 32 non-background pixels, bbox 6x6 | `BonusEnemyBig` vs `BonusEnemyInverse`: L1 202, 22 mismatch pixels, max abs 14 |
| direct fast luma64 | 12/12 | 29 non-background pixels, bbox 7x7 | `BonusEnemySlow` vs `BonusGameClear`: L1 174, 29 mismatch pixels, max abs 6 |

Plain read:

- Browser sprites are technically unique after grayscale/downsample at this
  position, but several are very close. The closest pair differs by only 22
  total gray levels over 7 pixels.
- Center-pixel identity is not enough for browser sprites. Example center luma
  collision: `BonusSelfFast` and `BonusEnemyStraightAngle` both had center value
  `59` in this probe.
- `circles_fast` is much more separated because it intentionally assigns
  type-coded fills: center values were `78, 92, 106, 120, 134, 148, 162, 176,
  190, 204, 218, 232`.
- The direct fast player-perspective path exposed a subtle collision risk:
  `BonusGameClear` starts as luma `232`, which is also player-1 head gray in a
  2P source row. The perspective remap rewrote it to `128` for controlled
  player 0 in this probe. That made it near `BonusEnemySlow` (`134`) and
  `BonusSelfMaster` (`120`) by value, even though the full patch hash stayed
  unique.

## Caveats

- This was one centered position, one source bonus radius (`3.0`), and no
  trails, heads, overlaps, edge clipping, natural spawn timing, or browser DOM
  canvas.
- The worktree was already dirty, including
  `src/curvyzero/env/vector_visual_observation.py`; this note treats the current
  file contents as the inspected state and does not edit production code.
- This is a signature probe, not a learning result. It only says what is
  visually separable under simple pixel metrics.

## Recommendation

An explicit opt-in `simple_symbols` renderer is reasonable as a speed/GPU
training experiment, but it needs its own contract and tests. It should not be
presented as parity with the trusted `browser_sprites` CPU reference.

Minimum tests before trusting it:

- Render all 12 source-default bonus types at the same position and require
  unique full-frame or crop hashes after the actual gray64 conversion used by
  training.
- Sweep several positions, including 11x11 downsample phase boundaries, edges,
  and partially clipped symbols; require no exact collisions and record minimum
  pairwise L1/mismatch-pixel margins.
- Include the player-perspective remap in the test path, or reserve symbol
  luma values that cannot be rewritten as body/head codes. Pin `BonusGameClear`
  specifically because luma `232` overlaps player-1 head gray.
- Test draw order with trails and live heads over/near bonuses, since production
  draws bonuses before heads.
- Keep the renderer explicitly opt-in and schema-labeled. The CPU
  `browser_sprites` path remains the reference; `simple_symbols` is an
  approximation/canary for training speed.

If bonus identity is important and grayscale symbols remain fragile, prefer the
existing semantic bonus64-style planes as the clean canary rather than encoding
more facts into one gray channel.
