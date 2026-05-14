# Bonus Symbol Preview

Date: 2026-05-14

Scope: standalone visual preview and critique only. No production code, Modal
jobs, or live runs were changed.

## Artifact

Preview image:
`artifacts/local/curvytron_render_profiles/bonus_simple_symbols_preview_20260514.png`

The preview uses the requested `simple_symbols` direction: 12 classes from
`3` outer masks x `4` thick internal marks, rendered as final grayscale
`64x64` patches. The sheet shows each symbol both as an enlarged `9x9` stamp
and as a centered `64x64` patch, plus a bottom strip of actual 1x `64x64`
patches.

Current local code state checked before making the artifact:
`src/curvyzero/env/vector_visual_observation.py` still exposes
`browser_sprites` and `circles_fast`; I did not find a production
`simple_symbols` mode in this worktree yet. This preview is therefore a design
artifact keyed to the main-thread contract, not an implementation parity claim.

## Mapping Previewed

- Codes `1-4`: diamond outer mask, base luma `116`, marks `N/E/S/W`.
- Codes `5-8`: circle outer mask, base luma `172`, marks `N/E/S/W`.
- Codes `9-12`: square outer mask, base luma `204`, marks `N/E/S/W`.
- Internal mark luma is `48`; background is `34`.

This is code-order grouping, not a perfect semantic grouping. The 12 active
bonus types have 4 self, 5 enemy, and 3 game/all classes, so any strict `3x4`
grid either splits the enemy family or leaves holes.

## Separability Read

Enlarged: all 12 classes are visually and mechanically separable. The outer
shape plus luma band gives strong row identity, and the thick cardinal marks
are easy to tell apart inside each row.

Small `64x64`: all 12 preview patches have unique hashes. The nearest preview
pairs are within the diamond/self row, where only the internal mark changes:

| pair | L1 | mismatch px | max abs |
| --- | ---: | ---: | ---: |
| `01` vs `02` | `1088` | `16` | `68` |
| `01` vs `04` | `1088` | `16` | `68` |
| `02` vs `03` | `1088` | `16` | `68` |
| `03` vs `04` | `1088` | `16` | `68` |

Plain critique: as a direct final-`64x64` symbol code, this looks much safer
than the current luma-only/circle-style bonus identity. The weak class is not
one specific bonus type; it is the same-row cardinal-mark distinction after
possible downsample phase shifts or occlusion.

## Caveats

- The preview stamps directly in final grayscale. If the implementation draws
  through the 704-style RGB canvas and then area-downsamples, run a phase sweep
  before trusting these margins.
- If production shrinks the footprint below this `9x9` stamp, the `N/E/S/W`
  marks may become the limiting factor.
- Edge clipping and partial overlap with trails or heads can still erase class
  identity. Full occlusion is allowed to destroy identity; partial occlusion
  should be measured.
- Keep `browser_sprites` as the reference/default. `simple_symbols` should stay
  explicit and metadata-labeled if added.

Recommendation: this symbol family is worth testing as the opt-in
`simple_symbols` renderer, with the first gate being exact `uint8` uniqueness
and minimum-distance sweeps over phase, edge clipping, radius, and draw-order
overlap.
