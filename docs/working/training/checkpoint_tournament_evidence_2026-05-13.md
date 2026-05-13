# Checkpoint Tournament Evidence, 2026-05-13

## Code Facts

- Current canonical training launcher:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- Current two-seat helper:
  `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- Current checkpoint eval loader:
  `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py`
- Current GIF browser:
  `src/curvyzero/infra/modal/curvytron_gif_browser.py`
- Current tournament Modal app:
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`

## Reuse Points

- `_find_state_dict` and `_make_policy_and_env` are the safest checkpoint loader
  path right now.
- `SourceStateGray64Stack4` renders per-player `[4,64,64]` observations from
  `VectorMultiplayerEnv`.
- `render_source_state_rgb_canvas_like` produces the rich RGB frames humans want
  to inspect.
- `run_management.clean_id`, `require_relative_ref`, `write_json`, and
  `file_ref` are good boring artifact helpers.
- The tournament Modal app defines its own image and volume constants instead
  of importing the full training launcher at module import time. That keeps the
  browser path lighter.

## Design Evidence

- The single-ego eval lane is not enough because it evaluates one checkpoint
  against a wrapper opponent.
- A real tournament needs two checkpoint policies in the same physical
  two-player env.
- The existing two-seat training loop already proves the direct env route:
  `VectorMultiplayerEnv(batch_size=..., player_count=2, ...)`.
- The existing live GIF worker proves a source-state RGB GIF can be produced
  without browser pixels.

## External Modal Evidence

- Modal functions support `.spawn()` for background execution and return a
  `FunctionCall` handle.
- Modal functions support `.map()` and `order_outputs=False` for parallel work
  where completion order is acceptable.
- Modal batch processing docs recommend `.spawn_map()` with `modal run --detach`
  for very large detached batches. We are not using it in V0 because pair and
  tournament functions need child results to write summaries.
- Modal Volume docs say changes need commit/background commit to become visible
  to other containers, and reload can fail when files are open. That matches
  the earlier website `open files preventing reload` error.

Sources checked:

- https://modal.com/docs/reference/modal.Function
- https://modal.com/docs/reference/modal.FunctionCall
- https://modal.com/docs/guide/batch-processing
- https://modal.com/docs/guide/volumes

## Storage Evidence

The existing training GIF browser uses a training-shaped layout:

`training/lightzero-curvytron-visual-survival/<run>/attempts/<attempt>/eval/<eval>/selfplay/...`

Tournament artifacts should not use that. They should use:

`tournaments/curvytron/<tournament_id>/battles/<battle_id>/games/<game_id>/...`

## Risk Evidence

- Checkpoint support sizes can drift; use the eval loader because it can infer
  model support from checkpoint weights.
- Greedy eval can hide training-style stochastic behavior; every game summary
  must record policy mode, temperature, and epsilon.
- Huge round robins can create many files; aggregate at pair/tournament levels
  and avoid shared writes from game workers.
- Concurrent Volume commits can contend. Game workers currently commit small
  immutable files; that is acceptable for smoke, but likely needs sharding for a
  very large tournament.
- Pair and tournament aggregates should be recomputable because workers can
  finish out of order.

## Current Test Evidence

- Unit tests cover pair construction, game spec construction, score extraction,
  standings recompute, ref validation, and local browser listing.
- Unit tests intentionally do not load real LightZero checkpoints.
- A real Modal smoke is still required before calling this production-ready.
