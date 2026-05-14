# Optimized Path Reorientation - 2026-05-13

Status: working memory, not a completion claim.
Owner surface: Environment wrapper/render contract.

Use this note to remember how the current fast path fits the environment work.
It does not replace the source-fidelity queue.

## Plain State

- The product runtime is still `VectorMultiplayerEnv`.
- The trainer-facing visual surface is
  `SourceStateMultiplayerTrainerSurface`.
- The trainer image is produced by `SourceStateGray64Stack4`.
- Default trainer images are not metadata and not ALE. They are source-state
  visual stacks: `float32[B,P,4,64,64]`.
- The default render mode is `browser_lines`.
- `body_circles_fast` is allowed only as an explicit approximate mode.
- `fast_gray64_direct` is rejected by the trainer surface. It remains a
  profile/custom path, not the default training observation.

## Current Observation Path

For the default path:

```text
VectorMultiplayerEnv state arrays
-> render source-state RGB canvas-like frame at 704x704
-> draw browser-line trails and browser sprite bonuses
-> convert RGB to luma
-> area-downsample 11x11 blocks to gray64
-> remap to player perspective
-> update the four-frame FIFO stack
-> expose policy rows only for live legal seats
```

For two-player rows, the stack uses the optimized helper
`render_source_state_canvas_gray64_player_perspectives(...)`. That helper is an
equivalence optimization: it renders/reuses shared trail work, uses a dirty
render cache when safe, then still emits the same declared `browser_lines`
gray64 tensor.

For three and four players, the stack currently falls back to per-player
renders. That is slower, but it keeps the same visual contract.

## What Recent Environment Fixes Feed

Recent lifecycle, leave, hit-owner, and bonus stack/death fixes mutate
`VectorMultiplayerEnv.state` and `batch.info`. The trainer image stack reads
that same state, so those fixes feed the same visual path.

Important distinction:

- Engine rule fidelity lives in `VectorMultiplayerEnv` and lower runtime code.
- Visual fidelity lives in the source-state renderer and stack.
- Optimizer can speed up the stack only when emitted tensors stay equivalent
  under the declared render mode.

## Metadata Added

`SourceStateMultiplayerTrainerSurface` step/reset info now exposes:

- `trainer_supported_trail_render_modes`;
- `visual_stack_dirty_render_stats`;
- render metadata that explicitly says browser-sprite bonuses are not an
  approximation in the default canvas-gray64 path.

This helps Optimizer and Coach read timing artifacts without guessing which
path produced an observation.

## Fresh Guard

Focused tests now pin that the two-player dirty render cache still matches a
full independent render when only an active bonus type changes at the same
position and radius. That protects the optimized path from reusing a stale
bonus sprite.

Trainer terminal tests now also compare the final stack frame against a direct
`render_source_state_canvas_gray64(...)` render of the same final
`VectorMultiplayerEnv.state`. That protects terminal `final_observation` rows
from drifting away from the declared source-state renderer.

## Still Open

- Full CurvyTron environment fidelity is not done.
- Browser pixel parity is not claimed.
- Real LightZero buffer sampled-target parity is downstream and still open.
- Wider 3P/4P render optimization is not done; current P=3/P=4 visuals are
  contract-correct but not optimized like the P=2 path.
- More source-backed multiplayer stress remains: broader leave variants,
  broader collision edges, and more bonus stack/death combinations.
