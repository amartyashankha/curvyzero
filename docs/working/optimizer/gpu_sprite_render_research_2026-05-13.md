# GPU Sprite Render Research

Date: 2026-05-13
Updated: 2026-05-15

Scope: historical research note for original bonus sprites on GPU.

## Current Status

This is not the current policy-observation recommendation.

The current production policy target is CPU `cpu_oracle`
`browser_lines + simple_symbols` with a `[4,64,64]` stack. GPU rendering is a
lab/profiling lane only until a batched trainer boundary exists and is profiled.

`browser_sprites` are for artifact/reference/browser-fidelity work. Use this
note only if that lane is explicitly reopened.

## Reusable Lessons

The useful GPU lessons still apply to the current target:

- batch real source-state rows;
- keep render output near policy/search when possible;
- avoid repeated host/device transfers;
- avoid Python loops in the hot path;
- compare GPU gray64 output against a CPU oracle before optimizing;
- measure transfer, render, readback, and handoff separately.

External anchors from the original research:

- NVIDIA texture guidance:
  https://developer.nvidia.com/docs/drive/drive-os/6.0.7/public/drive-os-linux-sdk/common/topics/graphics_content/Textures124.html
- CUDA best-practices guide:
  https://docs.nvidia.com/cuda/archive/8.0/cuda-c-best-practices-guide/index.html
- NVIDIA GPU Gems pipeline notes on alpha blending bandwidth:
  https://developer.nvidia.com/gpugems/gpugems/part-v-performance-and-practicalities/chapter-28-graphics-pipeline-performance
- JAX vectorization docs:
  https://docs.jax.dev/en/latest/automatic-vectorization.html
- PixelBrax paper for the broader "env plus renderer on GPU" shape:
  https://arxiv.org/abs/2502.00021

## Original Sprite Requirements

If original `browser_sprites` are ever needed on GPU, the faithful path must:

- upload the 3x4 RGBA atlas once;
- preserve source placement, clipping, nearest sampling, alpha blend, RGB
  rounding, BT.601 luma, and 11x11 downsample;
- preserve draw order: trails, sprites, heads;
- handle sprite overlap deterministically;
- compare bytes against CPU `browser_lines + browser_sprites` on real rows.

That is a separate artifact/reference project, not the policy-observation
target.

## Historical Toy Probe

The old scratch work showed why direct 64x64 sprite shortcuts are unsafe for
original sprites:

- exact per-pixel block accumulator: `0` mismatches in `250/250` placements;
- exact interval/count accumulator: `0` mismatches in `250/250` placements;
- naive 64x64 center sampling: `247/250` placements mismatched;
- luma-space blending before RGB rounding: `127/250` placements mismatched.

Plain read: original-sprite parity requires preserving 704-pixel RGB
alpha/rounding semantics or an exactly equivalent block-local computation. That
is unnecessary for the current `simple_symbols` policy target.

## Avoid

- Do not claim GPU sprite replacement until original-sprite parity passes.
- Do not let original-sprite parity block lab/profiling work on
  `browser_lines + simple_symbols`.
- Do not optimize around `body_circles_fast`; it is historical control only.
- Do not copy full RGB canvases to and from GPU per step.
- Do not call a short synthetic render probe a training speedup.
