# Browser Rendering Spec Handoff

Date: 2026-05-11

Purpose: define the training-observation rendering contract for CurvyTron visual
inputs. The goal is not browser theater. The goal is a cheap, correct board-fact
renderer that gives the model the same actionable game facts a player sees:
background occupancy, live heads, active map bonuses, clears/gaps, terminal
state, and stable layer order. Browser sprites, bounce animation, HUD icons,
death particles, idle arrows, and exact canvas antialiasing are useful only when
they encode game facts that the policy needs.

## Priority Framing

Highest priority:

- Trails must represent continuous occupied paths correctly.
- Live heads must be visible in the right place and color/luma class.
- Active map bonuses must be visible at the right location, radius, and type if
  type is exposed to the observation.
- Clears and explicit segment breaks must prevent false walls.
- Renderer metadata must say which approximation is active.

Lower priority unless proven useful for training:

- Bonus bounce-in animation.
- Bonus stack/HUD icons.
- Death/explosion particles.
- Idle arrows.
- Pixel-exact browser antialiasing, resize preservation, and DOM canvas timing.

The current renderer split is still the right shape:

- `browser_lines`: default source-state training renderer. It should draw
  connected rounded trail lines from `visual_trail_*` points when present, then
  fall back to sparse source/vector body state. It then draws active map bonuses
  from the source sprite atlas when `bonus_type` is available, then live heads.
- `body_circles_fast`: explicit cheap approximation that preserves the old
  circle-per-body raster. It may be useful for speed tests, but it must be
  labeled approximate.
- `browser_sprites`: default bonus renderer. It decodes
  `third_party/curvytron-reference/web/images/bonus.png` as the one 300x400 RGBA
  source atlas: 3 columns by 4 rows, 12 nominal 100x100 tiles. It maps source
  bonus type codes to browser tile indices and scaled-alpha-blits a cached
  per-tile/per-pixel-size stamp at `bonus_pos`/`bonus_radius`.
- `circles_fast`: explicit bonus fallback/profiling mode. It uses simple
  type-colored circles and is not the faithful path.

This document is therefore a replication plan for training observations, not a
promise of browser canvas pixel parity.

## Current Top Gaps

- Keep the raw/full-res/downsample contract pinned everywhere it exits the env:
  704x704 RGB source-state frame -> luma -> 11x11 area-downsampled gray64 ->
  stack.
- Keep continuous trails correct in the training renderer. `browser_lines`
  already prefers `visual_trail_*` points and connects sparse body points when
  visual trail points are absent; do not reintroduce dense-client gap heuristics
  on sparse body state.
- Keep typed active map bonuses visible through the source sprite atlas path.
  Gray64 visibility is a board fact, not a full hidden bonus-state contract.
- Keep wall/borderless/collision facts separate from pixels: collision truth is
  stored body circle overlap plus death metadata, not rendered line crossing.
- Browser canvas golden-frame parity is still later work.

## Current Visual State

- Bonus sprites come from one source atlas:
  `third_party/curvytron-reference/web/images/bonus.png`, 300x400 RGBA, 3x4,
  12 tiles.
- Atlas mapping is source code -> source name -> browser tile index:

| Code | Name | Tile |
| ---: | --- | ---: |
| 1 | `BonusSelfSmall` | 9 |
| 2 | `BonusSelfSlow` | 2 |
| 3 | `BonusSelfFast` | 0 |
| 4 | `BonusSelfMaster` | 5 |
| 5 | `BonusEnemySlow` | 3 |
| 6 | `BonusEnemyFast` | 1 |
| 7 | `BonusEnemyBig` | 6 |
| 8 | `BonusEnemyInverse` | 8 |
| 9 | `BonusEnemyStraightAngle` | 11 |
| 10 | `BonusGameBorderless` | 4 |
| 11 | `BonusAllColor` | 7 |
| 12 | `BonusGameClear` | 10 |

- Shared canvas-gray64 uses the raw 704x704 RGB source-state render, computes
  luma, then area-averages each 11x11 block into the 64x64 `uint8` tensor.
- Current `browser_lines` trail rendering prefers `visual_trail_*` points when
  present, then falls back to sparse persisted body points plus
  `body_break_before` where available. This is implemented and regression
  tested as the source-state training renderer, not as browser pixel parity.
- Source frames are about 16.67 ms. `BaseGame.framerate` is
  `(1 / 60) * 1000`.
- Source collision truth is stored body circle endpoint overlap plus
  death/collision metadata. A visual trail crossing alone is not source truth.
- Browser pixel parity is not claimed.

## Trail Topology Status

Regression rule: do not apply the browser client's `abs(dx) > 1` or
`abs(dy) > 1` gap rule directly to sparse persisted body points.

Why:

- Browser client trail points are dense. The browser receives every `position`
  event while printing and appends those points to the visual trail.
- Persisted server body points are sparse. The server only inserts a collision
  body when distance from the draw cursor is strictly greater than avatar radius.
- Applying the browser client's dense-point `>1` gap rule to sparse persisted
  body points turns normal trails into beads/disconnected segments.

Current rule for the source-state renderer:

- Prefer `visual_trail_*` points when present.
- Otherwise connect sparse stored body points for the same player in body order.
- Break only when explicit segment state exists: global clear, printing-off
  visual break, known wrap/teleport break, round reset, or
  `body_break_before`/future segment metadata.
- If explicit segment state is unavailable, prefer connecting sparse body points
  over inventing browser-client gaps. False beads are worse for training than a
  small amount of overconnection.

Regression priority: P0. This affects the main board fact: where walls are.

## Confirmed Frame Cadence And Collision Issue

Source CurvyTron schedules frames at about 16.67 ms through
`BaseGame.framerate = (1 / 60) * 1000`.

Collision is not defined by rendered line crossing. Source collision is based on
stored body circle endpoint overlap. Use death/collision metadata and stored
body geometry as source truth. Treat a visual trail crossing by itself as a
rendering clue only.

Trainer wrappers previously advanced one 300 ms decision as one physics step.
That can tunnel through stored bodies. Current source-frame wrappers use
`decision_source_frames` as the real knob, derive `decision_ms`, hold controls
for that window, simulate internally with source-sized frames, and stop early if
a row dies. The current default is 12 source frames, about 200 ms.

## Training Observation Matrix

| Piece | Current status | Source refs | Needed state | Implementation priority |
| --- | --- | --- | --- | --- |
| Trails | `browser_lines` is the default connected-line mode. It prefers `visual_trail_*` points when present and falls back to sparse persisted body points. It uses explicit break state when present. `body_circles_fast` remains explicit approximate mode. | Browser draws persistent rounded lines in `Game.drawTail()` and `Canvas.drawLineScaled()` (`third_party/curvytron-reference/src/client/model/Game.js:211`, `third_party/curvytron-reference/src/client/core/Canvas.js:304`). Server sparse body insertion is gated by `Avatar.isTimeToDraw()` (`third_party/curvytron-reference/src/server/model/Avatar.js:29`, `third_party/curvytron-reference/src/server/model/Avatar.js:40`). Client dense trail points come from `position` while printing (`third_party/curvytron-reference/src/client/model/Avatar.js:74`). | For training: prefer active `visual_trail_*` points, otherwise connect active persisted body points by owner/body order, using radius-derived width and deterministic color/luma. Consume explicit segment-break state for true clears, print-off gaps, wraps, and resets. Do not infer gaps from `>1` between sparse stored bodies. | P0. Main occupancy fact. |
| Collision truth | Confirmed source truth is stored body circle endpoint overlap plus death/collision metadata. Visual trail crossing alone is not enough. | Source collision uses stored body geometry; source frames run at about 16.67 ms from `BaseGame.framerate = (1 / 60) * 1000`. | Trainer wrappers now use `decision_source_frames`, derive `decision_ms`, advance source-sized internal frames, and stop early on death. Do not collapse a 300 ms decision into one physics step. | P0. One large step can tunnel through stored bodies and miss deaths. |
| Heads | Implemented as colored circles drawn after trails/bonuses. Good enough for board fact visibility, though not browser offscreen-canvas exact. | Browser head is rendered on avatar offscreen canvas and drawn on the front/game layer (`third_party/curvytron-reference/src/client/model/Avatar.js:149`, `third_party/curvytron-reference/src/client/model/Game.js:225`). | Use `present && alive`, current position, radius, and color/luma. Keep draw order above trails and map bonuses. | P0. Live head location is core policy state. |
| Bonus sprites | Implemented for the canvas-like path as the default `browser_sprites` bonus renderer. It decodes the one 300x400, 3x4 `web/images/bonus.png` atlas, caches scaled RGB/alpha stamps by tile index and pixel size, and falls back to simple circles only when `bonus_type` is missing or `bonus_render_mode="circles_fast"` is selected. | Browser sprite sheet is declared in `BonusManager` (`third_party/curvytron-reference/src/client/manager/BonusManager.js:14`, `third_party/curvytron-reference/src/client/manager/BonusManager.js:33`); map bonus radius comes from `BaseBonus` (`third_party/curvytron-reference/src/shared/model/BaseBonus.js:31`). | For training: draw active typed map bonus at `bonus_pos` with `bonus_radius` using the code -> name -> tile mapping above. Keep this source-state atlas path distinct from browser pixel parity. | P1 implemented for source-state canvas-like RGB/gray64; still not browser-pixel exact. |
| Bonus animation | Not replicated. Sprite glyph is static at full radius. This is fine unless animation conveys spawn timing or catchability beyond active state. | Browser `MapBonus` uses `BounceIn(300)` and computes animated draw size (`third_party/curvytron-reference/src/client/model/bonus/MapBonus.js:10`); manager updates and redraws it (`third_party/curvytron-reference/src/client/manager/BonusManager.js:96`). | For training: stable active/not-active visibility is enough. Add spawn-age animation only if experiments show the model needs time-since-spawn or browser catch timing. | P3. Cosmetic by default. |
| Bonus stack/HUD | Not rendered in board gray64. That is acceptable unless stacked bonuses are part of the observation contract. | Browser draws stack icons next to avatars on the game/front layer (`third_party/curvytron-reference/src/client/model/Game.js:195`, `third_party/curvytron-reference/src/client/model/Game.js:266`); server emits stack changes (`third_party/curvytron-reference/src/server/controller/GameController.js:436`). | Decide explicitly: either expose stack state through structured tensors/metadata or add compact board-side indicators. Do not add browser HUD art just for visual parity. | P2 if stacked bonuses affect control and are not otherwise exposed; P4 for browser icon parity. |
| Death/explosion effects | Not rendered. Acceptable for training after terminal unless final-frame visuals are used for value/reward diagnostics. | Browser pushes `Explode` animations on death (`third_party/curvytron-reference/src/client/model/Game.js:303`, `third_party/curvytron-reference/src/client/animation/Explode.js:7`). | For training: terminal/alive flags and final board occupancy matter more than particles. Add a simple terminal marker only if final observations are ambiguous. | P3. Usually non-actionable after death. |
| Idle arrows | Not rendered. Acceptable unless idle/local-player identity is part of the observation problem. | Browser draws local idle arrow on the effect layer (`third_party/curvytron-reference/src/client/model/Game.js:198`, `third_party/curvytron-reference/src/client/model/Game.js:285`; arrow canvas is built in `third_party/curvytron-reference/src/client/model/Avatar.js:163`). | For training: omit unless an agent must infer controllable/local seat from pixels. Prefer structured seat/channel metadata over decorative arrows. | P4. Not a board fact for current training. |
| Layer order | Mostly correct in native renderer: background/trails, active bonuses, heads. Missing optional HUD/effect overlays. | Browser DOM layers are `background`, `bonus`, `game`, `effect` (`third_party/curvytron-reference/src/client/views/game/play.html:102`), with z-index order in `_game.scss:439`. `Game.draw()` draws tails, bonus stacks, heads/arrows in browser flow (`third_party/curvytron-reference/src/client/model/Game.js:178`). | Training renderer should lock deterministic order: background fill, trails, map bonuses, live heads, optional stack markers, optional effects. Metadata should record omitted layers. | P0 for trails/bonus/head order; P3/P4 for optional overlays. |
| Antialiasing/resizing/persistence | Not browser-pixel exact. Native renderer rerenders from source/vector state into the raw 704x704 canvas-like RGB frame, then derives gray64 by 11x11 area downsampling. This is acceptable for training if board facts are stable and comparable. | Browser canvas scale is set on resize (`third_party/curvytron-reference/src/client/model/Game.js:315`); background can preserve pixels on resize (`third_party/curvytron-reference/src/client/core/Canvas.js:67`, `third_party/curvytron-reference/src/client/model/Game.js:311`). Browser line raster uses canvas antialiasing through `drawLineScaled()` (`third_party/curvytron-reference/src/client/core/Canvas.js:304`). | For training: render the same raw canvas-like layer used for GIF/human inspection, default 704x704, then convert luma and area-downsample each 11x11 block to 64x64. Direct 64x64 drawing is only an explicit profiling/legacy choice, not the default gray64 contract. Browser resize persistence is not needed for model input unless training replays browser UI resize events. | P1 for deterministic resize/luma/downsample and no topology artifacts; P4 for exact browser antialias/resize persistence. |

Priority key:

- P0: blocks useful visual training or creates wrong board facts.
- P1: important for fidelity once P0 is correct.
- P2: important only if the fact is not available elsewhere.
- P3: inspection/human/GIF polish or terminal diagnostics.
- P4: browser presentation parity with little expected policy value.

## Source-State Renderer Contract

`browser_lines` should mean:

1. Clear a fixed RGB frame to browser background `#222222`.
2. Construct per-player trail polylines from active `visual_trail_*` points when
   available.
3. Fall back to sparse persisted bodies by owner/body order.
4. Split only on explicit segment-break/clear state, not on the client dense
   `>1` gap heuristic.
5. Draw rounded trail strokes with width derived from body/avatar radius.
6. Draw active typed map bonus sprites over trails with `browser_sprites`.
7. Draw live present heads over map bonuses.
8. For gray64, render the raw canvas-like RGB layer first at the default
   browser-like source size, currently 704x704.
9. Convert that source RGB to luma, area-average each 11x11 block, and round to
   the 64x64 `uint8` training tensor.

`body_circles_fast` should mean:

1. Preserve the old circle-per-body raster.
2. Mark metadata as approximate.
3. Never describe it as browser-style trail fidelity.

`circles_fast` bonus rendering should mean:

1. Preserve a cheap circle/type-color bonus glyph for profiling and fallback.
2. Require explicit opt-in through `bonus_render_mode`.
3. Never treat grayscale or luma-coded circles as the faithful bonus path.

The renderer may be browser-inspired without being browser-pixel exact. Browser
pixel parity requires a separate harness with real canvas captures. That harness
is lower priority than correct training board facts.

## Required State

Current vector/source state is sufficient for a useful first-pass training
renderer:

- Player state: `present`, `alive`, `pos`, `radius`, `heading`, `printing`.
- Body state: `body_active`, `body_pos`, `body_radius`, `body_owner`,
  `body_num`, `body_insert_tick`, `body_insert_kind`, `body_break_before`,
  `body_write_cursor`, `body_count`, and `live_body_num`.
- Bonus state: `bonus_active`, `bonus_type`, `bonus_id`, `bonus_pos`,
  `bonus_radius`.
- Color state: `avatar_color` and `base_avatar_color`.

Missing or weak state that would improve correctness:

- `body_visual_segment_id` for grouping/replay convenience beyond the current
  per-body `body_break_before` continuity bit.
- Clear/reset epoch for replay renderers that reconstruct intermediate frames.
- Optional stack/active-effect facts if visual-only observations must include
  bonus stack effects.
- Browser DOM canvas captures/golden frames if true browser pixel parity becomes
  a goal later.

Do not block the 2P training observation on browser DOM captures. Visual trail
points plus sparse body connection with explicit breaks are the current
source-state training-observation fallback.

## Metadata Requirements

Every exported rendered frame or schema should expose enough metadata to prevent
silent fidelity confusion:

```python
"default_trail_render_mode": "browser_lines"
"supported_trail_render_modes": ["browser_lines", "body_circles_fast"]
"trail_render_mode": selected_mode
"trail_renderer_kind": "connected_rounded_lines" | "circle_per_body"
"trail_renderer_truth_level": (
    "source_state_browser_style_lines_non_pixel_parity"
    | "source_state_fast_body_circle_approximation"
)
"trail_renderer_is_approximation": selected_mode == "body_circles_fast"
"default_bonus_render_mode": "browser_sprites"
"supported_bonus_render_modes": ["browser_sprites", "circles_fast"]
"bonus_renderer_kind": "source_sprite_atlas_tiles" | "type_colored_circles"
"bonus_sprite_cache": "in_process_lru_stamp_cache_by_tile_index_and_pixel_size"
"bonus_sprite_atlas_size": [300, 400]
"bonus_sprite_tile_count": 12
"bonus_renderer_is_approximation": selected_bonus_mode == "circles_fast"
"rgb_source_frame_size": 704
"downsample_target_frame_size": 64
"downsample_method": "integer_area_average_after_luma"
"downsample_ratio": 11
"connects_sparse_persisted_bodies": selected_mode == "browser_lines"
"requires_explicit_segment_breaks": selected_mode == "browser_lines"
"browser_pixel_fidelity": False
"omitted_browser_layers": [
    "bonus_bounce_animation",
    "bonus_stack_hud",
    "death_particles",
    "idle_arrows",
]
```

The omission list is not an apology. It is a statement that these browser
presentation layers are lower priority for training unless they carry a missing
game fact.

## Tests To Add Or Keep

P0 trail topology tests:

- Sparse same-owner body points separated by more than one world unit must be
  connected in `browser_lines` when no explicit break state is present.
- Explicit break/clear state must split a trail when such state exists.
- `body_circles_fast` must preserve old bead output and mark itself approximate.
- Default RGB and gray64 mode must equal explicit `browser_lines`.
- Unknown `trail_render_mode` must raise `VectorVisualObservationError`.

P1 board-fact tests:

- Heads draw above trails and bonuses.
- Active typed bonus sprites draw at the right location/radius.
- Source bonus type codes map to the audited browser atlas tile indices.
- All 12 source default bonus types render distinct sprite patches.
- Gray64 sprite observations are the downsampled luminance of the high-resolution
  sprite-rendered RGB path, not direct 64x64 drawing or the circle fallback.
- Gray64 conversion preserves trail/head/bonus visibility at 64x64.
- Source snapshot and vector-state renders agree for the same selected mode.
- Renderer metadata records selected mode, approximation flag, and
  `browser_pixel_fidelity=False`.

Lower-priority tests, only when those features become intentional:

- Bonus bounce timing.
- Stack/HUD visual markers.
- Death particle overlays.
- Idle arrow overlays.
- Browser canvas golden-frame pixel comparisons.

## Browser Source Facts

Layer facts:

- Browser DOM layers are four stacked canvases: `background`, `bonus`, `game`,
  and `effect` (`third_party/curvytron-reference/src/client/views/game/play.html:102`).
- Sass places them in order: background z-index 30, bonus 40, game/front 50,
  effect 60 (`third_party/curvytron-reference/src/sass/pages/_game.scss:439`).
- Background starts as `#222222` and `clearBackground()` refills it
  (`third_party/curvytron-reference/src/client/model/Game.js:38`,
  `third_party/curvytron-reference/src/client/model/Game.js:293`).
- `clearTrails()` clears the browser background layer
  (`third_party/curvytron-reference/src/client/model/Game.js:125`).

Trail facts:

- `Game.drawTail()` draws `avatar.trail.getLastSegment()` onto the persistent
  background canvas with round caps
  (`third_party/curvytron-reference/src/client/model/Game.js:211`).
- `Canvas.drawLineScaled()` sets `lineCap`, `strokeStyle`, and scaled
  `lineWidth`, then `moveTo`/`lineTo`s the points
  (`third_party/curvytron-reference/src/client/core/Canvas.js:304`).
- Default avatar radius is `0.6`, so default browser trail width is `1.2`
  world units (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:54`).
- Client `Trail.addPoint()` has the `abs(dx) > 1` / `abs(dy) > 1` rule for
  dense browser trail points
  (`third_party/curvytron-reference/src/client/model/Trail.js:58`).
- Server `Avatar.update()` emits position frequently but inserts persisted
  bodies only when `isTimeToDraw()` passes
  (`third_party/curvytron-reference/src/server/model/Avatar.js:29`,
  `third_party/curvytron-reference/src/server/model/Avatar.js:40`,
  `third_party/curvytron-reference/src/server/model/Avatar.js:55`).
- Client receives `position`, sets avatar position, and appends a visual trail
  point while printing
  (`third_party/curvytron-reference/src/client/repository/GameRepository.js:189`,
  `third_party/curvytron-reference/src/client/model/Avatar.js:74`).
- Printing toggles emit important points and can clear the client visual trail
  (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:297`,
  `third_party/curvytron-reference/src/server/controller/GameController.js:328`,
  `third_party/curvytron-reference/src/client/repository/GameRepository.js:206`).

Head and bonus facts:

- Heads are drawn on the game/front canvas from avatar offscreen canvases
  (`third_party/curvytron-reference/src/client/model/Avatar.js:149`,
  `third_party/curvytron-reference/src/client/model/Game.js:225`).
- Browser bonus sprites come from `images/bonus.png`, with type order in
  `BonusManager` (`third_party/curvytron-reference/src/client/manager/BonusManager.js:14`,
  `third_party/curvytron-reference/src/client/manager/BonusManager.js:33`).
- Map bonuses animate through `MapBonus`/`BounceIn`, then draw via the bonus
  manager (`third_party/curvytron-reference/src/client/model/bonus/MapBonus.js:10`,
  `third_party/curvytron-reference/src/client/manager/BonusManager.js:96`).
- Bonus stack icons are drawn near avatars on the game/front layer
  (`third_party/curvytron-reference/src/client/model/Game.js:266`).

Effect facts:

- Death pushes an `Explode` animation onto the effect canvas
  (`third_party/curvytron-reference/src/client/model/Game.js:303`,
  `third_party/curvytron-reference/src/client/animation/Explode.js:7`).
- Local idle arrows also use the effect layer
  (`third_party/curvytron-reference/src/client/model/Game.js:285`).

## Validation Commands

Run the docs guard after editing this file:

```bash
python3 scripts/check_environment_doc_status.py docs/working/environment/browser_rendering_spec_2026-05-11.md
```

Useful renderer tests after code changes:

```bash
uv run pytest tests/test_vector_visual_observation.py -q
uv run pytest tests/test_compare_2p_raw_visual_observation.py -q
uv run pytest tests/test_curvyzero_source_state_visual_survival_lightzero_env.py tests/test_curvyzero_source_state_visual_turn_commit_lightzero_env.py -q
```

Do not use `scripts/compare_2p_raw_visual_observation.py --suite full2p` as a
browser pixel parity claim. It is a native/source-state consistency gate.
