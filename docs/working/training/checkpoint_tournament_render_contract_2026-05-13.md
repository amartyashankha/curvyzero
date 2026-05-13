# Checkpoint Tournament Render Contract, 2026-05-13

## Problem

Tournament GIFs currently look wrong. The visible symptom is that trails look
like straight-line approximations, and the output can drift away from the rich
CurvyTron view we want humans to inspect.

The root risk is simple: the tournament runner used one `trail_render_mode` for
two different jobs:

- the 64x64 stacked observation sent into the checkpoint policy
- the full-size GIF frame shown to humans

Those jobs should not share one knob.

## Contract

- Human GIFs always use the rich full-size render.
- Human GIFs are always `704x704` unless the canonical source renderer changes
  its full canvas size.
- Human GIFs use `browser_lines`, not `body_circles_fast` and not the direct
  gray64 path.
- Policy observations use the observation render mode that the checkpoint was
  trained with.
- If two seats need different policy observation modes, the tournament runner
  should keep separate observation stacks and feed each policy its own matching
  view.

## Current Limit

The Python rich renderer is browser-like, not a real browser canvas capture. It
uses source-state trail data and draws connected rounded line paths. That should
be good enough for inspection, but it is not proven pixel-identical to the
browser.

The main bug after the first split was not the rasterizer. The tournament game
runner was stepping the vector env as one large public decision step, so
`visual_trail_*` only got sparse decision-step points. The rich renderer then
connected those sparse points with straight segments.

Tournament games now use source-frame substeps:

- `decision_ms=200.0`
- `decision_source_frames=12`
- `source_physics_step_ms=16.666666666666668`
- `max_ticks=max_steps * decision_source_frames`

That makes one policy action hold across source-sized physics frames and gives
the GIF renderer dense trail points.

## Implementation Direction

- Keep old `trail_render_mode` as a backward-compatible alias for policy
  observations.
- Add `policy_trail_render_mode` for the model-facing observation path.
- Add `gif_trail_render_mode` in summaries/specs for clarity, but keep GIFs
  fixed to the rich browser-lines path for now.
- Store the actual modes in game summaries so the website and later audits can
  tell what happened.

## Validation

Local:

- `tests/test_curvytron_checkpoint_tournament.py`: 43 passed, 1 skipped.
- `tests/test_curvytron_two_seat_render_mode.py` and
  `tests/test_vector_visual_observation.py`: 67 passed, 1 skipped.
- Local source-frame probe appended 26 visual-trail points after one public
  two-player step.

Remote:

- `arena-render-contract-substep-smoke-20260513a`
- `elo-render-contract-substep-smoke`
- Two games completed.
- Pulled `game-000000/summary.json` and `game.gif`.
- Summary had `frame_size=704`, `gif_trail_render_mode=browser_lines`,
  `decision_source_frames=12`, and mixed policy observations:
  seat 0 `body_circles_fast`, seat 1 `browser_lines`.
- Pulled GIF was `704x704`, 16 frames, and visual inspection showed curved dense
  trails with a bonus visible.
