# Browser Rendering Spec Handoff

Date: 2026-05-11

Purpose: record the browser-style source-state renderer contract and the
remaining browser/canvas pixel parity gap. The trainer should ultimately see a
faithful grayscale version of the game view when speed allows. The old fast
circle-per-body renderer stays available, but it must be explicitly named as
approximate.

## Implementation Status

Current status: source-state renderer split implemented for native renders;
real browser/canvas pixel parity is still missing.

Renderer contract:

- `browser_lines`: the fidelity/default path. It should draw source-state
  browser-style connected rounded trail lines, then bonuses, then live heads.
  It is the default for RGB and gray64 source-state visual renders.
- `body_circles_fast`: the explicit approximation path. It preserves the old
  circle-per-body raster for speed tests, profiling, and legacy exact-pixel
  fixtures, but metadata and docs must label it as approximate.

The important boundary: `browser_lines` is a browser-style source-state/native
renderer. It is not a browser pixel parity claim. Exact browser/canvas parity
still needs a separate harness and golden browser canvas reference.

## Browser Canvas Coverage Matrix

This is the current truth, in plain terms. The renderer is closer to the game
view than the old bead renderer, but it is not a full browser canvas clone.

| Browser piece | Current renderer status | Training note |
| --- | --- | --- |
| Dark background | Implemented as flat `#222222`. | Fine for gray64. |
| Trail lines | Implemented as connected rounded lines from persisted source/vector body points. | Much closer than beads and now the default. Still not exact browser pixels because the browser also uses client position-event trail points and canvas antialiasing. |
| Trail clears/gaps | Breaks on the browser's `abs(dx)>1` or `abs(dy)>1` tolerance. Global clear state is covered through source/vector state. | Good enough for current source-state gate; exact client trail segment history may need more state later. |
| Player heads | Implemented as colored circles drawn after trails and bonuses. | Reasonable for model input. Not exact offscreen-canvas browser raster/antialiasing. |
| Active map bonuses | Implemented as simple yellow bonus circles with white outline. | This shows that a bonus exists and where it is. It is not the browser sprite sheet, bounce animation, or type-specific image. |
| Bonus type/status | Not encoded by gray64 except through the visible generic bonus glyph. Separate bonus64 diagnostic tensor checks active bonus type and post-catch status. | Do not claim gray64 tells the model every hidden bonus fact. |
| Bonus stack icons/HUD | Not rendered in the product gray64 board view. | Probably not part of the board observation unless we deliberately choose to add HUD-like information later. |
| Death explosion/effect layer | Not rendered. | Usually not needed for control after terminal state; final observation exists, but no explosion animation. |
| Idle arrow/effect layer | Not rendered. | Not needed for current training observation. |
| Browser sprite assets | Not loaded. | Human/GIF parity can improve later with `images/bonus.png`; training may not need this if simple glyphs work. |
| Canvas antialiasing/resizing/persistence | Not exact. Renderer rerenders from state into a fixed frame. | Fine for current source-state/native gate, not browser pixel parity. |
| Real browser pixels | Missing. | Needs a separate browser golden-frame harness before any exact browser parity claim. |

So the answer is: no, this is not just about browser lines, but browser lines
were the largest obvious visual bug. The current model path is a source-state
board render, not a full browser UI render. It covers the core board facts:
background, trails, heads, active map bonus location, and terminal/final frames.
It intentionally skips or simplifies browser presentation pieces such as bonus
sprites, HUD stack icons, explosion effects, idle arrows, and antialiasing.

Current code evidence: `TRAIL_RENDER_MODE_DEFAULT` is `browser_lines`;
`body_circles_fast` is a separate supported mode; RGB and gray64 renderer ids
include the selected mode; schema metadata keeps
`browser_pixel_fidelity=False`.
`scripts/compare_2p_raw_visual_observation.py --suite full2p` is therefore an
internal source-state/native visual consistency gate, not a browser canvas pixel
gate.

Validation evidence from 2026-05-11:

- `uv run pytest tests/test_vector_visual_observation.py tests/test_curvyzero_source_state_visual_survival_lightzero_env.py tests/test_compare_2p_raw_visual_observation.py -q`
  passed: 45 tests.
- `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_gif_browser.py -q`
  passed: 40 tests, 6 skipped.
- `uv run python scripts/compare_2p_raw_visual_observation.py --suite full2p --format plain`
  passed: `canvas_gray64=35/35`, `typed_bonus=12/12`, `final_obs=pass`,
  `canaries=2/2`, `mismatch_pixels=0`, `max_abs_diff=0.0`.
- `uv run python scripts/check_environment_doc_status.py docs/working/environment`
  passed.
- `git diff --check` passed.

## Implementation Tracking Checklist

Source-state/native renderer tasks:

- [x] Add renderer mode constants: `browser_lines` and `body_circles_fast`.
- [x] Add `trail_render_mode` to `render_source_state_rgb_canvas_like()`.
- [x] Add `trail_render_mode` to `render_source_state_canvas_gray64()`.
- [x] Implement `browser_lines` as connected rounded trail segments.
- [x] Preserve old circle-per-body behavior as `body_circles_fast`.
- [x] Add source snapshot helpers with the same mode selection.
- [x] Keep source snapshot and vector-state renderers mode-parity compatible.
- [x] Add metadata/schema fields for selected mode, default mode, supported modes,
  approximation flag, and browser pixel parity caveat.
- [x] Plumb selected mode into env reset info, step info, replay rows, and raw
  visual render metadata.

Still missing:

- [ ] Build a real browser/canvas pixel harness.
- [ ] Add golden browser canvas reference frames at fixed viewport, scale, colors,
  timing, and input state.
- [ ] Compare native/source-state renders to those browser canvas pixels before
  making any browser pixel parity claim.

Tests to add/run:

- [x] Default RGB mode equals explicit `browser_lines`.
- [x] Default gray64 mode equals explicit `browser_lines`.
- [x] Unknown mode raises `VectorVisualObservationError`.
- [x] `browser_lines` connects trail points with visible line pixels.
- [x] `body_circles_fast` preserves the old bead/circle output.
- [x] The two modes differ on a diagonal or long-segment fixture.
- [x] RGB-to-gray conversion respects the selected trail mode.
- [x] Source snapshot and vector state match under each selected mode.
- [x] Metadata/schema exposes selected trail mode and approximation flag.
- [ ] Browser canvas golden-reference parity tests.

Validation commands:

```bash
uv run pytest tests/test_vector_visual_observation.py -q
uv run pytest tests/test_compare_2p_raw_visual_observation.py -q
uv run pytest tests/test_curvyzero_source_state_visual_survival_lightzero_env.py tests/test_curvyzero_source_state_visual_turn_commit_lightzero_env.py -q
python3 scripts/check_environment_doc_status.py docs/working/environment/browser_rendering_spec_2026-05-11.md
```

Open questions, not blockers for first implementation:

- Do we need `body_visual_segment_id` or break flags to represent print gaps
  exactly?
- Should first `browser_lines` raster use binary coverage, supersampling, or a
  drawing library for antialiasing?
- Which cross-player draw order best matches browser persistence once browser
  screenshots are available?
- When should GIF/human render switch from simple bonus circles to the source
  sprite sheet?
- Is storing browser client position-event trail points worth the memory cost?

## Source Facts

Browser DOM layers are four stacked canvases:

- `background`: persistent trail/background canvas.
- `bonus`: map bonus canvas.
- `game`: front/head/avatar canvas.
- `effect`: death/arrow/effect canvas.

Refs:

- `third_party/curvytron-reference/src/client/views/game/play.html:102`
  creates `background`, `bonus`, `game`, and `effect` canvases in `#render`.
- `third_party/curvytron-reference/src/sass/pages/_game.scss:439` gives
  z-index `30` to background, `40` to bonus, `50` to game/front, and `60` to
  effect.
- `third_party/curvytron-reference/src/client/model/Game.js:43` binds the DOM
  canvases; bonus binds its own canvas in
  `third_party/curvytron-reference/src/client/manager/BonusManager.js:51`.

Layer behavior:

- Background starts as `#222222`:
  `third_party/curvytron-reference/src/client/model/Game.js:38`.
- `clearBackground()` fills the whole background canvas:
  `third_party/curvytron-reference/src/client/model/Game.js:293`.
- `repaint()` clears background, effect, and game/front, then draws:
  `third_party/curvytron-reference/src/client/model/Game.js:153`.
- `clearTrails()` only clears the background layer:
  `third_party/curvytron-reference/src/client/model/Game.js:125`.
- `onResize()` rescales all canvases and preserves the background image on
  resize through `setDimension(..., update=true)`:
  `third_party/curvytron-reference/src/client/model/Game.js:311` and
  `third_party/curvytron-reference/src/client/core/Canvas.js:67`.

Effect layer:

- Death pushes an `Explode` animation onto the effect canvas:
  `third_party/curvytron-reference/src/client/model/Game.js:303`.
- Local idle arrow also uses effect:
  `third_party/curvytron-reference/src/client/model/Game.js:285`.
- Current source-state renderer does not need effect pixels for model input.
  GIF/human render can add them later as a separate optional overlay.

## Trail Drawing

Browser trails are connected rounded strokes on the persistent background layer.
They are not bead/circle trails.

Draw call:

- `Game.drawTail()` gets `avatar.trail.getLastSegment()` and calls
  `background.drawLineScaled(points, avatar.width, avatar.color, 'round')`:
  `third_party/curvytron-reference/src/client/model/Game.js:211`.
- `avatar.width = avatar.radius * 2`:
  `third_party/curvytron-reference/src/client/model/Avatar.js:13`.
- Default radius is `0.6`, so default line width is `1.2` world units:
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:54`.
- `Canvas.drawLineScaled()` uses `lineCap = 'round'`, `strokeStyle = color`,
  `lineWidth = width * scale`, `moveTo(points[0] * scale)`, then `lineTo` for
  each later point:
  `third_party/curvytron-reference/src/client/core/Canvas.js:304`.
- The canvas wrapper scale is `render_width / game.size`:
  `third_party/curvytron-reference/src/client/model/Game.js:315`.

Trail point lifecycle:

- Base trail stores `points`, `lastX`, and `lastY`:
  `third_party/curvytron-reference/src/shared/model/BaseTrail.js:4`.
- Base `addPoint(x, y)` pushes `[x, y]` and updates `lastX/lastY`:
  `third_party/curvytron-reference/src/shared/model/BaseTrail.js:25`.
- Client `Trail.addPoint()` checks an axis tolerance of `1` world unit. If the
  previous point exists and `abs(dx) > 1` or `abs(dy) > 1`, it asks for a clear
  and queues the new point:
  `third_party/curvytron-reference/src/client/model/Trail.js:58`.
- `getLastSegment()` returns a copy of all queued points. If clear was asked, it
  clears the trail list and then starts a new list with the queued point. If no
  clear was asked and there are more than one point, it keeps only the last point
  after drawing:
  `third_party/curvytron-reference/src/client/model/Trail.js:28`.
- Client `Trail.clear()` only sets `clearAsked = true`; it does not clear the
  background canvas:
  `third_party/curvytron-reference/src/client/model/Trail.js:74`.
- Server/global clear clears collision bodies and emits `clear`; browser receives
  it and clears the background canvas:
  `third_party/curvytron-reference/src/server/model/Game.js:198`,
  `third_party/curvytron-reference/src/server/controller/GameController.js:498`,
  and `third_party/curvytron-reference/src/client/repository/GameRepository.js:319`.

Important subtlety:

- Browser client trail points are not the same thing as persisted collision
  bodies.
- Server emits `position` on every server position update:
  `third_party/curvytron-reference/src/server/model/Avatar.js:55` and
  `third_party/curvytron-reference/src/server/controller/GameController.js:340`.
- Client receives `position`, sets avatar position, and if `printing` is true it
  adds a trail point immediately:
  `third_party/curvytron-reference/src/client/repository/GameRepository.js:189`
  and `third_party/curvytron-reference/src/client/model/Avatar.js:74`.
- Server only inserts persisted collision bodies when `isTimeToDraw()` passes,
  which means no draw cursor exists or distance from cursor is strictly greater
  than avatar radius:
  `third_party/curvytron-reference/src/server/model/Avatar.js:29` and
  `third_party/curvytron-reference/src/server/model/Avatar.js:40`.
- Important point events are a second client trail source. Printing toggles call
  `addPoint(x, y, true)` and clear the client visual trail when printing turns
  off:
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:297`.
  Server `Avatar.addPoint()` emits the `important` flag:
  `third_party/curvytron-reference/src/server/model/Avatar.js:158`.
  The server controller forwards only important `point` events:
  `third_party/curvytron-reference/src/server/controller/GameController.js:328`.
  The browser receives that `point` event and adds the current avatar point:
  `third_party/curvytron-reference/src/client/repository/GameRepository.js:206`.
- Therefore connecting vector `body_*` points is a real improvement over beads,
  but it is still a source-state browser-style renderer, not proven browser pixel
  parity. Exact parity needs either client trail-point reconstruction or browser
  output validation.

## Head And Avatar Drawing

Heads are drawn on a separate front canvas over trails and bonuses.

- `Avatar.drawHead()` clears the avatar offscreen canvas and fills a circle with
  `radius * scale` at the offscreen center:
  `third_party/curvytron-reference/src/client/model/Avatar.js:149`.
- `Avatar.update()` computes `startX` and `startY` as rounded scaled position
  minus offscreen canvas radius:
  `third_party/curvytron-reference/src/client/model/Avatar.js:56`.
- Browser rounding is `(0.5 + value) | 0`:
  `third_party/curvytron-reference/src/client/core/Canvas.js:340`.
- `Game.drawAvatar()` draws the offscreen head canvas to the front layer and
  records the rectangle for the next clear:
  `third_party/curvytron-reference/src/client/model/Game.js:225`.
- `Game.draw()` clears prior head rectangles before drawing updated heads:
  `third_party/curvytron-reference/src/client/model/Game.js:178`.
- Dead or absent avatars are not redrawn unless marked changed:
  `third_party/curvytron-reference/src/client/model/Game.js:180`.

Renderer mapping: draw heads after trails and map bonuses. Use `present && alive`
for current vector-source model input unless a future replay mode deliberately
renders death animation/effect frames.

## Bonus Drawing

Browser map bonuses are sprites on the `bonus` canvas:

- Sprite source is `images/bonus.png`, arranged as 3 by 4:
  `third_party/curvytron-reference/src/client/manager/BonusManager.js:14`.
- Type-to-sprite order is listed at
  `third_party/curvytron-reference/src/client/manager/BonusManager.js:33`.
- `MapBonus` has `radius = 3` inherited from `BaseBonus`:
  `third_party/curvytron-reference/src/shared/model/BaseBonus.js:31`.
- It animates with `BounceIn(300)` and derives `drawRadius`, `drawWidth`,
  `drawX`, and `drawY` on update:
  `third_party/curvytron-reference/src/client/model/bonus/MapBonus.js:10`.
- The bonus manager clears the old zone, updates the animation while not done,
  and draws the sprite with `drawImageScaled`:
  `third_party/curvytron-reference/src/client/manager/BonusManager.js:96`.

For our current source-state renderer, it is enough to draw active map bonuses
from `bonus_active`, `bonus_pos`, `bonus_radius`, and optionally `bonus_type`.
The current yellow/white circle glyph is acceptable only as source-state
browser-like, not browser sprite parity. A later human/GIF renderer can load the
sprite sheet.

## Current Vector State

The vector runtime has the state needed for both source-state renderer modes:

- Player state: `present`, `alive`, `pos`, `radius`, `heading`, `printing`.
  See `src/curvyzero/env/vector_multiplayer_env.py:3454`.
- Trail state: `visible_trail_count`, `has_visible_trail_last`,
  `visible_trail_last_pos`, `has_draw_cursor`, `draw_cursor_pos`.
  See `src/curvyzero/env/vector_multiplayer_env.py:3528`.
- Body state: `body_active`, `body_pos`, `body_radius`, `body_owner`,
  `body_num`, `body_insert_tick`, `body_insert_kind`, `body_write_cursor`,
  `body_count`, and `live_body_num`.
  See `src/curvyzero/env/vector_multiplayer_env.py:3536`.
- Bonus state is added when seeded/natural bonuses are enabled:
  `bonus_active`, `bonus_type`, `bonus_id`, `bonus_pos`, `bonus_radius`.
  See `src/curvyzero/env/vector_multiplayer_env.py:1870` and
  `src/curvyzero/env/vector_multiplayer_env.py:2006`.
- Avatar color indices are `avatar_color` with `base_avatar_color`; default is
  `0..P-1` per row:
  `src/curvyzero/env/vector_multiplayer_env.py:2044`.

Current renderer facts:

- `render_source_state_rgb_canvas_like()` defaults to `browser_lines`; explicit
  `body_circles_fast` keeps the old circle-per-body behavior.
- `render_source_state_canvas_gray64()` renders RGB at 64 by 64 with the
  selected trail mode and converts to luma.
- Schema says this is source-state backed and not browser pixel parity.
- `scripts/compare_2p_raw_visual_observation.py --suite full2p` compares source
  snapshots and `VectorMultiplayerEnv` through these native/source-state render
  paths. It does not read browser canvas pixels.

## Two Renderer Paths

Define two paths and keep both.

Recommended constants:

```python
TRAIL_RENDER_MODE_BROWSER_LINES = "browser_lines"
TRAIL_RENDER_MODE_BODY_CIRCLES_FAST = "body_circles_fast"
TRAIL_RENDER_MODES = frozenset(
    (TRAIL_RENDER_MODE_BROWSER_LINES, TRAIL_RENDER_MODE_BODY_CIRCLES_FAST)
)
```

Recommended public API:

```python
def render_source_state_rgb_canvas_like(
    state,
    *,
    row=0,
    out=None,
    frame_size=SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    player_rgb=None,
    background_rgb=SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
): ...

def render_source_state_canvas_gray64(
    state,
    *,
    row=0,
    out=None,
    rgb_out=None,
    player_rgb=None,
    background_rgb=SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
): ...
```

Recommended private helpers:

```python
def _render_source_state_rgb_browser_lines(...): ...
def _render_source_state_rgb_body_circles_fast(...): ...
def _browser_line_polylines_from_body_state(...): ...
def _draw_rounded_world_polyline_rgb(...): ...
def _draw_world_circle_rgb_fast(...): ...
```

Snapshot helpers should mirror the same mode:

```python
def render_source_snapshot_rgb_canvas_like(..., trail_render_mode="browser_lines"): ...
def render_source_snapshot_canvas_gray64(..., trail_render_mode="browser_lines"): ...
```

Mode semantics:

- `browser_lines`: browser-style fidelity path. Draw connected rounded trail
  strokes on a virtual background layer. Use this for raw human visibility, GIFs,
  and the default candidate for model grayscale if profiling is acceptable.
- `body_circles_fast`: old circle-per-body approximation. Preserve it for speed
  tests, profiling, and deliberate low-cost input. It must never be described as
  browser-style fidelity.

Default:

- `render_source_state_rgb_canvas_like()` should default to `browser_lines`
  because the function name says browser-like and browser trails are connected
  rounded lines.
- `render_source_state_canvas_gray64()` should inherit the same default and pass
  the selected mode into the RGB renderer before luma conversion.
- Existing tests that expect the old bead image should pass
  `trail_render_mode="body_circles_fast"`.

## Browser-Lines Mapping Plan

Start with full-state rerender, not a persistent mutable canvas object. Each
render call should:

1. Create or clear an RGB frame to background `#222222`.
2. Build trail polylines from active body slots up to `body_write_cursor`.
3. Draw polylines with round caps and width `body_radius * 2` in world units.
4. Draw active bonus glyphs on top of trails.
5. Draw live present heads on top of bonuses.
6. Convert to gray64 when requested.

Polyline construction from current vector state:

- Filter `body_active[row, :body_write_cursor[row]]`.
- Group by `body_owner`.
- Within a player, use `body_num` order, with slot order as a stable tie-break.
  `body_num` is the source `AvatarBody.num`; see
  `third_party/curvytron-reference/src/server/core/AvatarBody.js:12`.
- Draw the player path as one or more polylines. Use each body's own radius for
  line width; if radius changes inside a player path, split at the width change.
- Break a line when browser client tolerance would break:
  `abs(dx) > 1` or `abs(dy) > 1`.
- Break line continuity for known visual clears. Current state is not ideal here:
  `body_insert_kind` helps, but it does not by itself fully identify print-start
  versus print-stop visual boundaries. A future `body_visual_segment_id` or
  `body_break_before/body_break_after` field would remove this ambiguity.

Ordering caution:

- Browser background persistence means draw order is time-based across frames.
- A full-state renderer cannot exactly replay that order unless the state carries
  enough event history or body insertion timing.
- `body_num` is per player, not global. `body_write_cursor` slot order is global
  insertion order, but grouping by player is needed to draw continuous lines.
- First implementation should choose a deterministic order and label it. A good
  candidate is reverse player order to match `Game.draw()` avatar loop
  (`third_party/curvytron-reference/src/client/model/Game.js:186`) while sorting
  points by `body_num` inside each player. Browser parity tests may later force a
  different order.

Source snapshot mapping:

- `CurvyTronSourceEnv.world_bodies_snapshot()` exposes persisted collision
  bodies:
  `src/curvyzero/env/source_env.py:725`.
- `bonus_bodies_snapshot()` exposes active map bonuses:
  `src/curvyzero/env/source_env.py:745`.
- `avatar_body_metadata_snapshot()` exposes radius/body counters:
  `src/curvyzero/env/source_env.py:760`.
- These snapshots still do not expose every browser client position-event trail
  point. Browser-lines from snapshots should carry the same caveat as vector
  state.

## Metadata And Schemas

Expose the selected mode everywhere a rendered frame can leave the renderer.

Recommended schema fields:

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
"browser_trail_semantics": "persistent_background_canvas_round_line_caps"
"browser_client_trail_point_caveat": (
    "vector/source snapshots expose persisted body points, not all client "
    "position-event trail points"
)
"browser_pixel_fidelity": False
"browser_pixel_fidelity_claim": "not_validated_against_browser_canvas"
```

Renderer ids should include the mode:

```python
SOURCE_STATE_RGB_BROWSER_LINES_RENDERER_IMPL_ID = (
    "curvyzero_source_state_rgb_canvas_like_browser_lines_numpy/v0"
)
SOURCE_STATE_RGB_BODY_CIRCLES_FAST_RENDERER_IMPL_ID = (
    "curvyzero_source_state_rgb_canvas_like_body_circles_fast_numpy/v0"
)
SOURCE_STATE_CANVAS_GRAY64_BROWSER_LINES_RENDERER_IMPL_ID = (
    "curvyzero_source_state_canvas_gray64_browser_lines_numpy/v0"
)
SOURCE_STATE_CANVAS_GRAY64_BODY_CIRCLES_FAST_RENDERER_IMPL_ID = (
    "curvyzero_source_state_canvas_gray64_body_circles_fast_numpy/v0"
)
```

Where to record it:

- `source_state_canvas_gray64_schema()` should name the default and supported
  modes.
- Any metadata helper should include the selected `trail_render_mode`.
- `last_reset_info`, `timestep.info`, replay rows, profile rows, and raw visual
  render metadata should include `trail_render_mode`, `trail_renderer_kind`, and
  `trail_renderer_truth_level`.
- The fast path should set an explicit approximate flag. The browser-line path
  should set a browser-style flag but still keep `browser_pixel_fidelity=False`
  until validated against browser output.

## Tests To Add

Keep old fast renderer behavior:

- A fixture with two or three active body points should match the old
  circle-per-body RGB/gray output when
  `trail_render_mode="body_circles_fast"`.
- Existing source snapshot vs vector visual tests should be updated to pass
  `body_circles_fast` if they assert exact old pixels.
- Schema/metadata test must assert fast mode says approximate.

Prove browser-line mode exists and differs:

- A diagonal two-point trail should render connected non-background pixels
  between endpoints in `browser_lines`.
- The same fixture should leave the middle empty or different in
  `body_circles_fast`.
- `render_source_state_canvas_gray64(..., trail_render_mode="browser_lines")`
  should equal `rgb_canvas_like_to_gray64(render_source_state_rgb_canvas_like(...,
  trail_render_mode="browser_lines"))`.
- Repeat the same luma equality for `body_circles_fast`.

Prove default choice:

- Calling `render_source_state_rgb_canvas_like()` without mode should equal an
  explicit `trail_render_mode="browser_lines"` call.
- Calling `render_source_state_canvas_gray64()` without mode should also equal
  explicit `browser_lines`.
- Schema should report `default_trail_render_mode == "browser_lines"`.

Prove browser clear/gap cautions are handled as far as current state allows:

- A body sequence with `abs(dx) > 1` or `abs(dy) > 1` should split into separate
  line segments in `browser_lines`.
- A sequence with a future segment marker, if added, should not connect across
  that marker.
- If implementation only has `body_insert_kind`, add a test documenting the
  chosen conservative break rule and keep the browser point caveat in metadata.

Prove API rejects ambiguity:

- Unknown `trail_render_mode` should raise `VectorVisualObservationError`.
- Public env config that selects `body_circles_fast` should surface that exact
  mode in `last_reset_info` and step info.

Candidate files:

- `tests/test_vector_visual_observation.py`
- `tests/test_compare_2p_raw_visual_observation.py`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`
- `tests/test_curvyzero_source_state_visual_turn_commit_lightzero_env.py`

## Gaps And Cautions

- Persistent browser canvas vs full-state rerender: browser only draws new trail
  segments to a persistent background canvas. Our renderer will likely rerender
  a snapshot from arrays. This is useful and deterministic, but not the same
  execution model.
- Body points vs client trail points: current vector/source snapshots expose
  persisted collision body points. Browser trails also include position-event
  points while printing and important client-side boundary points. Connected body
  lines are better than beads but still need validation.
- Trail gaps: materialized bodies remain during visual holes. Do not connect all
  bodies blindly. Need segment breaks for print toggles, wraps, and clear events.
- Wraps: client tolerance clears the trail if the next point jumps by more than
  `1` in x or y, which prevents drawing a long line across borderless wrap.
- Clear events: `BonusGameClear` clears server world bodies and browser
  background. A source-state renderer after the clear should only see post-clear
  bodies, but replay renderers need event order if rendering intermediate frames.
- Ordering: `body_num` orders points per owner. Slot order records insert order.
  Browser draw order is persistent and frame-based. Be explicit about the chosen
  order until browser comparisons pin it down.
- Source snapshot vs vector state: both are source-backed, but neither is a DOM
  canvas dump.
- Browser pixel parity: source-state browser-like rendering is not browser pixel
  parity until compared against real browser canvas pixels at fixed inputs,
  sizes, colors, and timing.
- Bonus sprites: current source-state renderer can keep simple bonus circles.
  Human/GIF parity should eventually use `images/bonus.png`.
- Antialiasing: browser canvas strokes are antialiased. A NumPy line renderer
  may be binary unless we add coverage sampling or use a raster library. Metadata
  should say what it does.

## Implemented Shape

The source-state/native renderer split now exists:

- `src/curvyzero/env/vector_visual_observation.py` owns the mode constants,
  validation, RGB renderers, gray64 conversion, schema metadata, and renderer ids.
- `browser_lines` is the default for source-state RGB and gray64.
- `body_circles_fast` preserves the old circle-per-body approximation as an
  explicit mode.
- Source snapshot and vector-state renderers use the same mode selection.
- Source-state visual env metadata carries selected/default/supported trail mode
  and keeps browser pixel fidelity false.

Config can still select the source-state mode explicitly:

```python
source_state_trail_render_mode = "browser_lines"
```

- For any debug/profiling path selecting the fast mode, record:

```python
trail_render_mode = "body_circles_fast"
trail_renderer_is_approximation = True
```

Remaining validation commands:

```bash
uv run pytest tests/test_vector_visual_observation.py -q
uv run pytest tests/test_compare_2p_raw_visual_observation.py -q
uv run pytest tests/test_curvyzero_source_state_visual_survival_lightzero_env.py tests/test_curvyzero_source_state_visual_turn_commit_lightzero_env.py -q
python3 scripts/check_environment_doc_status.py docs/working/environment/browser_rendering_spec_2026-05-11.md
```

Missing browser-pixel work:

- Capture browser canvas golden frames at fixed source state, viewport, scale,
  colors, and timing.
- Compare native/source-state output to those browser pixels with an explicit
  tolerance policy.
- Keep that harness separate from
  `scripts/compare_2p_raw_visual_observation.py --suite full2p`.

## Later Profiling, Not A Blocker

Profile the source-state renderer modes separately from browser pixel work:

- 64 by 64 gray64 render time for both modes.
- 704 by 704 RGB render time for GIF/human inspection.
- Batch throughput for `browser_lines` vs `body_circles_fast`.
- Cost of antialiasing or supersampling.
- Cost of grouping/sorting by owner/body number.
- Whether storing client trail points or segment ids is cheaper than
  reconstructing from body arrays.

Do not block browser/canvas parity design on these profiles. The source-state
renderer split is useful now, but real browser pixel parity still needs a
separate browser harness and golden reference frames.
