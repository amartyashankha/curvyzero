# CurvyTron Render Trajectory Profile

Date: 2026-05-13

Purpose: quantify how much local CurvyTron env time is render time when
trajectories are forced to run long.

## Current Truth

The current trusted Coach training lane is stock LightZero:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train
env_variant=source_state_fixed_opponent
```

For this stock lane, the render comparison is:

- `browser_lines`: richer browser-like source-state render.
- `body_circles_fast`: faster approximation used in the current live matrices.

Do not use the old `fast_gray64_direct` name for current stock fixed-opponent
runs. That name belongs to the superseded custom two-seat adapter.

Current Coach batches verified in docs/manifests pair both render modes:
`curvy-survive-bonus-large-20260513b`, `curvy-mix2-clean-20260513a`, and
`curvy-mix3-currentckpt-20260513a`.

## Measurement

Tool:

```text
scripts/profile_curvytron_render_trajectory_lengths.py
```

The tool now splits scalar full-render time from cached/perspective render time
and records dirty-cache stats in JSON/JSONL cells for future reruns.

Latest artifact after the stock fixed-opponent dirty-cache landing:

```text
artifacts/local/curvytron_render_profiles/render_trajectory_lengths_dirty_scalar_20260513.json
artifacts/local/curvytron_render_profiles/render_trajectory_lengths_dirty_scalar_20260513.cells.jsonl
```

Latest artifact after the direct fast gray64 hot-path landing:

```text
artifacts/local/curvytron_render_profiles/render_trajectory_lengths_directfast_20260513b.json
artifacts/local/curvytron_render_profiles/render_trajectory_lengths_directfast_20260513b.cells.jsonl
```

Previous scalar full-redraw baseline used for the comparison table:

```text
artifacts/local/curvytron_render_profiles/render_trajectory_lengths_20260513c.json
```

Run:

```text
uv run python scripts/profile_curvytron_render_trajectory_lengths.py \
  --lengths 100 200 500 1000 2000 \
  --render-modes browser_lines body_circles_fast \
  --repeats 1 \
  --warmup-steps 20 \
  --output artifacts/local/curvytron_render_profiles/render_trajectory_lengths_dirty_scalar_20260513.json \
  --markdown
```

Scope:

- local env-only profile;
- current `source_state_fixed_opponent` env wrapper;
- includes the exact dirty/incremental cache when the wrapper enables it;
- `death_mode=profile_no_death`;
- `opponent_runtime_mode=blank_canvas_noop`;
- wall-avoidant ego action heuristic;
- no LightZero search, learner, replay, checkpoint, eval, or GIF work;
- one clean pass per cell.

This profiles the fixed-opponent LightZero wrapper. It does not measure the
newer multiplayer trainer surface.

## Recent Environment Review

Rerun date: 2026-05-13 after recent environment commits.

What changed that can affect these numbers:

- runtime bonus/death/collision behavior changed, so `vector_step_sec`,
  terminal timing, bonus density, and long no-death state evolution can move;
- default bonus rendering is now browser-sprite based inside the full RGB
  canvas path;
- render mode names did not change for the stock fixed-opponent path:
  `browser_lines` and `body_circles_fast` are still the current knobs.

This answers the render Amdahl question for long env rollouts. It does not by
itself predict full training wall time when games are short or when MCTS/search
dominates.

## Latest Direct Fast Hot Path

Date: 2026-05-13, after the stock wrapper started using
`render_source_state_gray64_fast_player_perspectives` for
`body_circles_fast` model observations.

This is still local env-only, no-death, no LightZero search/learner. It answers
only the renderer/env wrapper question.

| render | steps | wall s | steps/s | render s | render % | observation s | vector step s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `browser_lines` | 100 | 0.533 | 187.5 | 0.399 | 74.7% | 0.414 | 0.069 |
| `body_circles_fast` direct gray64 | 100 | 0.065 | 1547.7 | 0.009 | 13.6% | 0.012 | 0.035 |
| `browser_lines` | 500 | 1.390 | 359.8 | 1.070 | 77.0% | 1.104 | 0.190 |
| `body_circles_fast` direct gray64 | 500 | 0.341 | 1467.8 | 0.060 | 17.6% | 0.078 | 0.177 |
| `browser_lines` | 1000 | 2.804 | 356.7 | 2.115 | 75.5% | 2.185 | 0.410 |
| `body_circles_fast` direct gray64 | 1000 | 0.799 | 1251.7 | 0.165 | 20.7% | 0.205 | 0.400 |
| `browser_lines` | 2000 | 5.802 | 344.7 | 4.373 | 75.4% | 4.519 | 0.859 |
| `body_circles_fast` direct gray64 | 2000 | 1.722 | 1161.6 | 0.451 | 26.2% | 0.527 | 0.818 |

Plain read:

- The direct fast approximation is now much faster than browser-lines in this
  local no-death lens: about `8.2x` at 100 steps and `3.4x` at 2000 steps.
- The old local result where `body_circles_fast` lost at long trajectories is
  historical. That row rendered the fast approximation through the full RGB
  path; the current hot path goes straight to gray64.
- After this patch, the fast approximation is no longer render-dominated at
  short horizons. At long horizons, vector/game stepping becomes comparable.

## Browser Lines, Dirty Cache On

| steps | wall s | steps/s | render s | render % | observation s | vector step s | other s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.046 | 95.6 | 0.548 | 52.4% | 0.564 | 0.448 | 0.498 |
| 200 | 3.129 | 63.9 | 1.469 | 46.9% | 1.495 | 1.543 | 1.660 |
| 500 | 10.492 | 47.7 | 5.195 | 49.5% | 5.245 | 5.109 | 5.297 |
| 1000 | 23.293 | 42.9 | 12.121 | 52.0% | 12.229 | 10.782 | 11.172 |
| 2000 | 46.903 | 42.6 | 26.898 | 57.3% | 27.096 | 19.277 | 20.004 |

Plain read: the exact dirty cache is a real win. `browser_lines` is still the
largest single bucket in long no-death env-only rollouts, but it is no longer an
overwhelming 88-89% redraw problem. The remaining local Amdahl ceiling from
render-only work is now roughly 2.1x-2.3x for 500-2000 step rows.

## Historical Body Circles RGB Path

| steps | wall s | steps/s | render s | render % | observation s | vector step s | other s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.637 | 156.9 | 0.364 | 57.2% | 0.368 | 0.251 | 0.273 |
| 200 | 2.000 | 100.0 | 1.321 | 66.0% | 1.330 | 0.628 | 0.679 |
| 500 | 13.063 | 38.3 | 8.299 | 63.5% | 8.326 | 4.610 | 4.765 |
| 1000 | 43.862 | 22.8 | 32.969 | 75.2% | 33.026 | 10.564 | 10.893 |
| 2000 | 86.656 | 23.1 | 63.664 | 73.5% | 63.797 | 22.208 | 22.992 |

Plain read: this table is now historical. It measured `body_circles_fast`
through the old RGB/downsample path before the direct gray64 hot path existed.
Do not use it as the current speed read.

## Comparison

| steps | old browser full-redraw wall | browser dirty-cache wall | dirty-cache speedup | body-circles wall | fastest in this local row |
| ---: | ---: | ---: | ---: | ---: | --- |
| 100 | 1.133s | 1.046s | 1.08x | 0.637s | `body_circles_fast` |
| 200 | 5.210s | 3.129s | 1.66x | 2.000s | `body_circles_fast` |
| 500 | 39.144s | 10.492s | 3.73x | 13.063s | `browser_lines` |
| 1000 | 96.774s | 23.293s | 4.15x | 43.862s | `browser_lines` |
| 2000 | 175.943s | 46.903s | 3.75x | 86.656s | `browser_lines` |

The render-only ceiling is the Amdahl limit if render became free and every
other local env cost stayed the same. It is not a promise; it tells us the
largest possible win from only attacking render in this narrow local regime.

## Optimizer Read

For long-lived policies, render remains a high-leverage target, but the first
big local win is now landed for the trusted rich visual path. The later direct
gray64 hot path also makes `body_circles_fast` much faster in env-only
no-death profiles. That speed result is real, but it is an approximation
surface, not a CPU-reference parity claim.

For current live training, do not overread this as "render is the whole training
bottleneck." The live stock runs include MCTS/search, learner work, subprocess
collection, checkpoint eval/GIF, and many short episodes. Existing live cadence
reads show `body_circles_fast` and `browser_lines` checkpoint gaps are often
closer than this env-only table, because full training has other costs.

Next useful optimizer steps:

1. Keep paired `browser_lines` and approximation rows in Coach matrices until
   training quality decides the fidelity tradeoff.
2. Prefer cached `browser_lines` whenever the question is CPU-reference
   fidelity. Prefer explicit approximation modes only when the question is
   speed/fidelity tradeoff.
3. The next renderer work should measure dirty-cache hit/fallback/dirty-block
   counts and check whether fixed-opponent can avoid rendering two identical
   player-perspective frames.
4. Run one full stock LightZero profile at a mature long-survival checkpoint
   later, so the env-only Amdahl table can be reconciled with real search and
   learner timing.
