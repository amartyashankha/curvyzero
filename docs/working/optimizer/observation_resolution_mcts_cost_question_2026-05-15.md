# OPEN QUESTION: Observation Resolution vs MuZero/MCTS Cost

Date: 2026-05-15

Do not let this disappear: should CurvyTron keep the current `64x64`
LightZero observation, or test larger tensors such as `[4,96,96]` and
`[4,128,128]` so trails, heads, and bonus symbols survive the final model grid
with more visible cells?

This is not settled. Treat it as an optimizer tradeoff: visual information
versus renderer cost, model cost, replay/IPC size, and MCTS/search throughput.
Do not claim policy quality improves until learning runs prove it.

## Correction: What `block_704_gray64` Means

The current GPU `block_704_gray64` prototype is full-resolution-aware inside
the renderer. It is not a direct 64x64 shortcut.

It does not materialize a full `704x704` RGB image. Instead, for each final
`64x64` output cell, it checks the high-resolution sample positions that
correspond to the old `704 -> 64` downsample block. Bonus `simple_symbols` are
also evaluated at those high-resolution sample positions, then averaged down
with the rest of the frame.

So the current model contract is still `[4,64,64]`, but trail/body pixels before
the final downsample are not being reduced to one direct sample per output cell.
The remaining question is whether the final model grid itself should stay 64.

## Current Wiring

- `source_state_fixed_opponent` is 2-player source-state visual training.
- Wrapper policy observation is a normalized grayscale FIFO stack:
  `[4,64,64]`.
- Local env/model config patches LightZero Atari MuZero to
  `model_type="conv"`, `image_channel=4`, `frame_stack_num=1`, and
  `observation_shape=[4,64,64]`.
- The inherited upstream Atari MuZero config has `downsample=True`.
- The CPU browser-like path can still render a `704x704` RGB canvas and
  downsample it with an `11x11` integer area average.
- The GPU block path should be understood as the cheaper equivalent contract:
  compute final 64 cells using the corresponding high-res samples.

## What Scales With Observation Size

Pixels in the stacked observation scale as `4 * H * W`:

- `64x64`: baseline.
- `96x96`: 2.25x more input pixels.
- `128x128`: 4.0x more input pixels.

This directly affects observation buffers, env/collector payloads, subprocess
IPC, replay storage, learner batches, and the root model `initial_inference`.

In MuZero/LightZero, MCTS does not run the representation network every
simulation. Verified in LightZero v0.2.0:

- `_forward_collect` runs `initial_inference(observation)` once for the ready
  roots, then calls MCTS search with the returned latent roots.
- `initial_inference` runs representation on the real image observation, then
  prediction.
- MCTS loops over `num_simulations` and calls `recurrent_inference` on latent
  states.
- `recurrent_inference` runs dynamics plus prediction. It does not read the
  image observation and does not rerun representation.

That means a larger image always makes the root representation pass more
expensive, but it affects every MCTS simulation only if the latent tensor after
representation is larger or if the prediction/reward heads get larger.

For stock LightZero conv MuZero with `downsample=True`, local config inherits
the Atari branch. LightZero v0.2.0 has explicit 64 and 96 observation branches:
64 is downsampled to an 8x8 latent; 96 is downsampled to a 6x6 latent after an
extra pooling step. A plain 128x128 observation is not a minimal config-only
change in that source; the model code does not have a matching 128 branch.

If `downsample=False` were used instead, latent spatial size would stay tied to
the input size, so recurrent MCTS cost would scale much more directly with
`H*W`. Do not mix those cases in conclusions.

## Geometry: Trail Width at Current Scale

The default trusted path is 2-player source-state. The exact code path is:

- `VectorMultiplayerEnv(..., player_count=2)` in the wrapper.
- `CurvyTronReferenceDefaults.avatar_radius = 0.6`.
- `CurvyTronReferenceDefaults.arena_size_for_players(2) = 88`, so
  `map_size = 88`.
- `VectorMultiplayerEnv` writes `state["radius"] = 0.6`; body and visual-trail
  radii are copied from that player radius.
- `browser_lines` calls `_draw_browser_line_trails_rgb`, then
  `_draw_rounded_world_polyline_rgb`.
- `_draw_rounded_world_polyline_rgb` uses
  `scale = canvas_size / map_size` and `radius_px = radius * scale`.

For the current 2-player geometry:

| Surface | Radius | Full width |
| --- | ---: | ---: |
| `704x704` source samples | `4.80 px` | `9.60 px` |
| `64x64` final grid | `0.44 cells` | `0.87 cells` |
| `96x96` final grid | `0.65 cells` | `1.31 cells` |
| `128x128` final grid | `0.87 cells` | `1.75 cells` |

At `64x64`, each output cell represents an `11x11` block of the 704-scale
source samples. A centered trail crossing a cell can still produce a strong
averaged luma signal because the source trail is about 9.6 pixels wide inside
an 11-pixel block. But the final line is still less than one model cell wide,
so 64 can blur sub-cell position, nearby parallel trails, thin turns, and exact
head/trail separation.

Larger observations would not add more source truth by themselves; they would
give the final model tensor more cells to represent the same geometry. For this
trail width, 96 moves the trail to about 1.3 cells wide, and 128 to about 1.75
cells wide. That may preserve shape and separation better, at the cost of more
observation pixels and possibly more model work.

Bonus `simple_symbols` are drawn in source-canvas coordinates and then
downsampled with the rest of the frame. With default bonus radius `3.0` and
`map_size=88`, their visible footprint in the final tensor is still roughly 3
cells at 64, 4 at 96, and 5 at 128 if the renderer is parameterized cleanly.
Current code is named and wired around gray64, so larger outputs still need an
explicit schema change.

## Support Reality

`96x96` and `128x128` are not current CLI toggles. The code and metadata are
named around gray64, with hard-coded `64`, `63`, `[4,64,64]`, and 704-to-64
integer downsampling in multiple paths.

Smallest credible support work:

1. Add an explicit observation-size field or explicit variants such as
   `gray64`, `gray96`, `gray128`.
2. Parameterize visual observation shapes, wrapper stack shape, scratch buffers,
   player-perspective buffers, normalization buffers, schema/hash metadata, and
   final-observation checks.
3. Parameterize final-grid coordinate math now written as `64` and `63`.
4. Pick render geometry:
   `704 -> 64`, `768 -> 96`, and `1024 -> 128` keep integer block semantics;
   `704 -> 96/128` would need a non-integer resampler; a direct block-aware GPU
   renderer can avoid materializing any raw canvas.
5. Update LightZero env/model observation shape. Use 96 first because stock
   LightZero has a 96 branch. Treat 128 as a model-code experiment, not a pure
   config experiment.
6. Make the observation schema/hash and checkpoint compatibility label change
   loudly.

## Concrete Experiment Plan

Do not touch live runs or Modal volumes while adding support.

1. Keep `64x64` as the control.
2. Add local-only shape/schema/visual probes for `64`, `96`, and `128` if the
   renderer/trainer supports them. Check reset/step smoke, player perspective,
   trail geometry, head/trail separation, bonus symbol clipping, and final
   observation shape.
3. First profile candidate after review: `96x96`, same
   `source_state_fixed_opponent`, `browser_lines + simple_symbols`, reward,
   opponent, batch size, collectors, eval cadence, and `sim8`.
4. Stress rows only after 96 works: `64/96/128` at `sim8`, then repeat the
   likely practical winner at `sim16`.
5. Profile `env steps/s`, wall time, observation render/packing time,
   subprocess IPC, replay push/sample time, learner time, GPU memory,
   `model_initial_inference_sec`, `model_recurrent_inference_sec`,
   `mcts_search_sec`, root batch size, and MCTS node budget.
6. Change the observation contract only if the larger grid clearly fixes a
   visual information problem, such as ambiguous bonus/head/trail pixels, while
   keeping throughput acceptable and without forcing smaller batches or root
   widths.

Current recommendation: keep `[4,64,64]` as the production contract until this
profile exists. If visibility is the only problem, cheaper alternatives such as
stronger final-grid symbols or extra semantic planes may beat a global
resolution increase.

## First Renderer-Only Probe

2026-05-15 H100 isolated GPU render rows, all real-env rollout state,
`browser_lines + simple_symbols`, B64, trail_slots128, real_env_steps128,
controlled_player0, readback enabled, verification skipped because production
CPU references are currently gray64-only:

| output | frame/source samples | device render | end-to-end with readback |
| --- | ---: | ---: | ---: |
| `64x64` | `704` | `14.80ms` | `18.35ms` |
| `96x96` | `768` | `15.62ms` | `18.89ms` |
| `128x128` | `1024` | `26.33ms` | `29.47ms` |

Plain read: renderer cost alone does not rule out `96x96`; it was nearly the
same as `64x64` in this isolated H100 row. `128x128` is meaningfully slower but
not catastrophic for the renderer. This does **not** answer the real question,
because model/root inference, replay/IPC size, and LightZero config support are
still unmeasured. Keep `64x64` as the launch contract and test `96x96` as the
first larger-observation experiment after the GPU backend question is cleaner.
