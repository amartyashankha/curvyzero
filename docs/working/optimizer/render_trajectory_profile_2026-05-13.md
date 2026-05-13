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

Artifact:

```text
artifacts/local/curvytron_render_profiles/render_trajectory_lengths_20260513b.json
artifacts/local/curvytron_render_profiles/render_trajectory_lengths_20260513b.cells.jsonl
```

Run:

```text
uv run python scripts/profile_curvytron_render_trajectory_lengths.py \
  --lengths 100 200 500 1000 2000 \
  --render-modes browser_lines body_circles_fast \
  --repeats 1 \
  --warmup-steps 20 \
  --output artifacts/local/curvytron_render_profiles/render_trajectory_lengths_20260513b.json \
  --markdown
```

Scope:

- local env-only profile;
- current `source_state_fixed_opponent` env wrapper;
- `death_mode=profile_no_death`;
- `opponent_runtime_mode=blank_canvas_noop`;
- wall-avoidant ego action heuristic;
- no LightZero search, learner, replay, checkpoint, eval, or GIF work;
- one clean pass per cell.

This answers the render Amdahl question for long env rollouts. It does not by
itself predict full training wall time when games are short or when MCTS/search
dominates.

## Browser Lines

| steps | wall s | steps/s | render s | render % | observation s | vector step s | other s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.367 | 73.1 | 1.045 | 76.4% | 1.052 | 0.287 | 0.323 |
| 200 | 6.278 | 31.9 | 5.502 | 87.6% | 5.518 | 0.699 | 0.776 |
| 500 | 37.781 | 13.2 | 35.164 | 93.1% | 35.202 | 2.434 | 2.617 |
| 1000 | 135.577 | 7.4 | 122.661 | 90.5% | 122.758 | 12.426 | 12.916 |
| 2000 | 233.842 | 8.6 | 217.328 | 92.9% | 217.558 | 15.470 | 16.514 |

Plain read: in this long no-death env-only regime, `browser_lines` is render
dominated almost immediately. At 500+ steps, about 90% or more of the local env
wall time is inside the gray64 render call.

## Body Circles Fast

| steps | wall s | steps/s | render s | render % | observation s | vector step s | other s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.002 | 99.8 | 0.624 | 62.3% | 0.631 | 0.341 | 0.378 |
| 200 | 2.989 | 66.9 | 1.869 | 62.5% | 1.893 | 0.978 | 1.119 |
| 500 | 13.843 | 36.1 | 10.840 | 78.3% | 10.883 | 2.803 | 3.002 |
| 1000 | 46.509 | 21.5 | 36.661 | 78.8% | 36.738 | 9.468 | 9.848 |
| 2000 | 68.480 | 29.2 | 56.690 | 82.8% | 56.825 | 11.153 | 11.791 |

Plain read: `body_circles_fast` is much faster than `browser_lines`, but it is
still render dominated in long no-death env-only rollouts. At 1000-2000 steps,
about 79-83% of local env wall time is still render.

## Comparison

| steps | browser wall | fast wall | fast speedup | browser render % | fast render % | browser render-only ceiling | fast render-only ceiling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.367s | 1.002s | 1.37x | 76.4% | 62.3% | 4.2x | 2.7x |
| 200 | 6.278s | 2.989s | 2.10x | 87.6% | 62.5% | 8.1x | 2.7x |
| 500 | 37.781s | 13.843s | 2.73x | 93.1% | 78.3% | 14.4x | 4.6x |
| 1000 | 135.577s | 46.509s | 2.92x | 90.5% | 78.8% | 10.5x | 4.7x |
| 2000 | 233.842s | 68.480s | 3.41x | 92.9% | 82.8% | 14.2x | 5.8x |

The render-only ceiling is the Amdahl limit if render became free and every
other local env cost stayed the same. It is not a promise; it tells us the
largest possible win from only attacking render in this narrow local regime.

## Optimizer Read

For long-lived policies, render remains a high-leverage target. The richer
`browser_lines` path has a large possible render-only win. The fast path is
already better, but still has a meaningful render ceiling.

For current live training, do not overread this as "render is the whole training
bottleneck." The live stock runs include MCTS/search, learner work, subprocess
collection, checkpoint eval/GIF, and many short episodes. Existing live cadence
reads show `body_circles_fast` and `browser_lines` checkpoint gaps are often
closer than this env-only table, because full training has other costs.

Next useful optimizer steps:

1. Keep paired `browser_lines` and `body_circles_fast` rows in Coach matrices
   until training quality decides the fidelity tradeoff.
2. For renderer optimization, focus on `body_circles_fast` first if it remains
   a serious training surface; it still spends most long-env time in render.
3. For richer visual fidelity, `browser_lines` needs a more structural render
   fix before it can be cheap in long-survival regimes.
4. Run one full stock LightZero profile at a mature long-survival checkpoint
   later, so the env-only Amdahl table can be reconciled with real search and
   learner timing.
