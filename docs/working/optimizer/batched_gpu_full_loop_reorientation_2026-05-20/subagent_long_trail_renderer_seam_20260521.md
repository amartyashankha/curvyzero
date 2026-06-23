# Long-Trail Renderer Seam Audit

Date: 2026-05-21

Scope: profile-only observation-renderer inspection for the current GPU/browser-lines source-state paths. This does not propose production behavior changes or live-training changes.

## Bottom Line

The long-run slowdown is in the observation renderer's full trail redraw, not in live training, search, or action masking. The latest dynamic GPU renderer profile at B512/A16 with a no-death trajectory ladder shows scalar steps/s falling from about 3976 at 20 steps to about 598 at 500 steps; at 500 steps, observation is about 96% of wall time and renderer render is about 94%.

The exact GPU hot seam is the per-frame loop over packed trail slots in the JAX block renderer:

- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py::_jax_render_block_704_gray64_two_views`
- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py::_jax_render_block_704_gray64`

Both redraw every selected trail slot into a fresh 704-equivalent block buffer every observation frame. Dynamic slot selection reduces the loop width, but once a trajectory is long it still does O(render_trail_slots per frame) work. In browser-lines mode there is also an O(trail_slots) previous-owner prepass.

The cleanest narrow seam for an incremental/dirty or cheaper long-trail experiment is a new profile-only `SourceStateBatchedObservationRenderer` implementation, wired through `source_state_batched_observation_boundary_profile.py` / the renderer-backed profile facade. Do not change the production stack, policy defaults, tournament renderers, or live trainer observation path.

## Where The O(active_trails per frame) Work Is

### GPU Dense Block Renderer

Primary two-view profile hot path:

- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py::_jax_render_block_704_gray64_two_views`
- It allocates a fresh image buffer with shape roughly `(2, B, 64, 64, 11, 11)`.
- In browser-lines mode it first calls `_previous_owner_trail_slots`.
- It then runs `lax.fori_loop(0, state["trail_x"].shape[1], draw_trail_slot, ...)`.
- `draw_trail_slot` computes cap or segment coverage for each slot over the full block-grid, applies owner priority/luma, and writes into the fresh image.
- Bonus sprites/simple symbols and heads are drawn after the trails by `_draw_block_bonus_and_heads_two_views`.

Single-view sibling:

- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py::_jax_render_block_704_gray64`
- Same core loop over `state["trail_x"].shape[1]`, with image shape roughly `(B, 64, 64, 11, 11)`.

Direct 64x64 sibling:

- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py::_jax_render_direct_gray64`
- It avoids the 11x11 block buffer, but still computes a broadcast hit tensor against all trail slots and all 64x64 pixels.
- Browser-lines mode still uses `_previous_owner_trail_slots` and `_segment_hits`.

Browser-lines prepass:

- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py::_previous_owner_trail_slots`
- Scans all trail slots to find each slot's previous active slot for the same owner and compatible radius.
- This is O(trail_slots) before the draw loop and matters once render slots are large.

### CPU Browser-Lines Renderer

Full RGB/gray scalar path:

- `src/curvyzero/env/vector_visual_observation.py::render_source_state_rgb_canvas_like`
- `src/curvyzero/env/vector_visual_observation.py::render_source_state_canvas_gray64`
- `src/curvyzero/env/vector_visual_observation.py::_render_source_state_rgb_browser_lines`
- `src/curvyzero/env/vector_visual_observation.py::_draw_browser_line_trails_rgb`
- `src/curvyzero/env/vector_visual_observation.py::_draw_ordered_browser_line_path_rgb`
- `src/curvyzero/env/vector_visual_observation.py::_draw_rounded_world_polyline_rgb`
- `src/curvyzero/env/vector_visual_observation.py::_draw_rounded_pixel_segment_rgb`

This path consumes `visual_trail_*` arrays when present. `_render_source_state_rgb_browser_lines` finds active slots, groups by owner draw order, sorts slots, then draws each owner path. `_draw_rounded_world_polyline_rgb` walks every adjacent point pair and calls `_draw_rounded_pixel_segment_rgb`, whose pixel work is proportional to each segment's bounding box. That is the CPU browser-lines full-redraw seam.

Two-player CPU dirty cache already exists:

- `SourceStateBrowserLineTrailLayerCache`
- `SourceStateCanvasGray64DirtyRenderCache`
- `_source_state_dirty_blocks_for_append`

These are useful as a behavioral reference for append-only trail caching and dirty block recomposition, but they are CPU-side, two-player-oriented, and not the current B512/A16 dynamic GPU profile path.

### Profile Boundary Around The Hot Renderer

The current profile-only boundary that prepares GPU render input is:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py::_render_candidate_frames_from_production_state`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py::_DynamicJaxBatchedObservationRenderer.render`

That path does:

1. Convert production source-state arrays to compact render arrays through `_production_to_benchmark_source_state`.
2. Pack trails into owner draw order through `_pack_compact_trails_in_owner_draw_order`.
3. Pick dynamic render width through `_select_render_trail_slots`.
4. Truncate compact arrays through `_truncate_compact_trails_for_render`.
5. Copy to device.
6. Call the JAX render function.
7. Copy frames back and convert view-major to row-major.

Important detail: `_DynamicJaxBatchedObservationRenderer.render` always renders the full batch from `request.state`; if the request is partial, it gathers requested rows after the full render. That keeps the current implementation simple, but it is not a true dirty/partial renderer.

## State And Data Structures Consumed

Production source state consumed before compacting:

- `visual_trail_active`
- `visual_trail_write_cursor`
- `visual_trail_pos`
- `visual_trail_radius`
- `visual_trail_owner`
- `visual_trail_break_before`
- Fallback body arrays when visual trails are unavailable
- `player_position`, `player_radius`, `alive`
- `avatar_color`
- bonus position/radius/type/active arrays

Compact GPU render state:

- `trail_x`
- `trail_y`
- `trail_radius`
- `trail_owner`
- `trail_active`
- `trail_break_before`
- `trail_write_cursor`
- `head_x`
- `head_y`
- `head_radius`
- `head_alive`
- `bonus_x`
- `bonus_y`
- `bonus_radius`
- `bonus_active`
- `bonus_type`
- `avatar_color`

Perspective/request data:

- `SourceStateBatchedRenderRequest.state`
- `row_indices`
- `controlled_players`
- `out`
- `trail_render_mode`
- `bonus_render_mode`

CPU dirty-cache state, useful as the existing exact append-only model:

- last write cursor
- cached visual-trail prefix signature
- cached map size and color key
- per-owner RGB trail layers
- owner occupancy masks
- cached current RGB/gray frames
- previous head snapshots
- previous bonus snapshots
- dirty 64x64 block mask

## Cleanest Profile-Only Seam

The clean seam is to add a new renderer implementation behind the existing `SourceStateBatchedObservationRenderer` protocol, used only by the renderer-backed profile harness. The natural insertion points are:

- `source_state_batched_observation_boundary_profile.py::_DynamicJaxBatchedObservationRenderer`
- `source_state_batched_observation_boundary_profile.py::_render_candidate_frames_from_production_state`
- `source_state_batched_observation_profile.py::SourceStateBatchedObservationProfileFacade`
- `multiplayer_source_state_trainer_surface.py::_RendererBackedSourceStateGray64Stack4`
- `source_state_hybrid_observation_profile.py::_update_observation` and `_reset_autoreset_observation`

Recommended shape:

- Keep the existing renderer unchanged as the baseline.
- Add a separate profile-only renderer, for example `incremental_dirty` or `cheap_long_trail`, selected by an explicit profile config/label.
- Let that renderer retain per-row cache state keyed by stable global row.
- Reset or rebuild on row reset, cursor regression, map-size change, palette/change in controlled-player semantics, trail prefix mutation, or unsupported bonus mode.
- Report telemetry for cache hits, full rebuilds, dirty-block count, active slots, render width, reset rows, and fallback reasons.

This should not touch:

- live `SourceStateGray64Stack4` defaults
- policy observation backend defaults
- tournament rendering
- checkpoint policy metadata semantics
- browser/GIF RGB renderer behavior

### Incremental/Dirty Renderer Option

The existing CPU dirty renderer is the best exact reference. It already demonstrates the right invalidation model:

- full rebuild on cold start
- full rebuild on cursor regression
- full rebuild on prefix mutation
- append only new visual-trail slots
- mark dirty blocks for new caps/segments, old and current heads, and bonus changes
- recompose only dirty blocks
- redraw bonuses and heads after trail layer composition

A GPU/profile version should use the same rules, but live inside a new profile renderer. The core goal is to avoid clearing and redrawing all long-trail slots into a fresh frame every step.

### Cheaper Long-Trail Representation Option

A cheaper representation can be prototyped at the compact-render boundary instead of inside production state:

- before `_pack_compact_trails_in_owner_draw_order`
- during owner-order packing
- after packing but before `_truncate_compact_trails_for_render`
- inside a new JAX renderer that accepts a different compact schema

Examples include recent-K exact segments plus a cached/coarse occupancy layer, low-resolution trail accumulation, or endpoint/spline simplification. These are not observation-equivalent to current browser-lines and must carry a distinct profile/render schema label.

## Risks And Contract Edges

### Terminal And Autoreset

The no-death ladder is useful for isolating long-trail cost, but it hides terminal/autoreset hazards. Normal death can reset rows, reuse row cache, and regress visual-trail cursors. Final observations must be produced from the pre-autoreset state, while reset-stack observations must be produced from the post-autoreset state.

`source_state_hybrid_observation_profile.py` already separates `render_state` and `autoreset_render_state`. Any incremental cache must honor that split, reset affected row caches, and not accidentally render final observations from reset state. Partial reset rendering is especially important because the current dynamic GPU renderer still full-renders the batch for partial requests.

### Player Perspective

The current GPU two-view block path is effectively specialized to two controlled players. It depends on `avatar_color`, player luma mapping, owner ordering, and row-major/view-major conversion. It must preserve the policy's per-player perspective contract, including the self/opponent luma relationship.

The scalar `_render_state_view` path also masks blank-canvas noop opponents from both body and visual-trail arrays. A compact or cached renderer must not resurrect hidden opponent trails when a perspective intentionally omits them.

### Bonus Sprites And Draw Order

Production RGB supports browser sprite bonuses, circles, and simple symbols. The GPU profile path is only safe for the policy modes it explicitly supports, especially `simple_symbols`. An incremental renderer must preserve draw order:

1. background
2. trails
3. bonuses
4. heads

Dirty rendering also has to redraw current bonuses when a dirty trail block intersects them, even if the bonus itself did not change.

### Action Masks

Action masks are not renderer inputs. They are carried by the environment/actor payload separately from observation frames. A renderer optimization must not infer liveness or legal actions from rendered pixels, and it must not change done/reset/action-mask packaging.

### Training And Tournament Observation Contracts

Current policy and tournament contracts are tied to observation shape, dtype, render mode, bonus mode, player perspective, and checkpoint metadata. A cheaper long-trail representation is a different observation surface unless proven exactly equivalent.

Do not promote a profile-only approximation into:

- live training defaults
- tournament evaluation
- source-state RGB/GIF renderers
- checkpoint-compatible policy observation metadata

without a separate contract update and validation pass.

### Approximation Risk

Line simplification, recent-K rendering, coarse occupancy, or trail accumulation may improve throughput but can change what the policy sees near collisions, old walls, bonuses, and player heads. That can invalidate comparisons against policies trained on browser-lines observations.

## Recommendation

For the next narrow experiment, keep the current dynamic GPU renderer as the baseline and add a second profile-only renderer implementation behind `SourceStateBatchedObservationRenderer`. Start with an exact dirty/incremental design modeled on `SourceStateCanvasGray64DirtyRenderCache`, because that attacks the measured O(long trail per frame) cost without changing the observation contract.

If exact incremental GPU rendering proves too costly to implement quickly, prototype a cheaper long-trail representation only in the boundary/profile path and label it as a distinct observation schema. Measure it separately from the browser-lines renderer and do not use it to claim production or tournament parity.
