# GPU Render Parity Gap

Date: 2026-05-13

Purpose: plain record of what the GPU render prototype matches, what it does
not match, and whether the mismatch looks meaningful.

## Target

The parity target here is the current CPU reference renderer:

```text
render_source_state_canvas_gray64(..., trail_render_mode="browser_lines")
```

That means:

- source-state RGB-style 704 canvas;
- browser-line-like trail geometry;
- live heads;
- browser-sprite bonus rendering in the CPU path;
- BT.601 luma;
- 11x11 area average downsample to uint8 `[1,64,64]`.

This is not a browser-canvas pixel claim. It is CPU-reference gray64 parity.

## History

1. **Direct64 toy**
   The first GPU probe rendered simple synthetic circles directly into 64x64.
   It hit exact parity against its own toy CPU renderer and was very fast, but
   it did not prove anything about the production browser-lines render.

2. **Early block-704 prototype**
   The next GPU path tried to approximate the production 704-to-64 render. The
   first version sampled world-space centers. It was visibly wrong at line
   edges because the CPU reference effectively works in source-pixel blocks
   before downsampling.

3. **Pixel-coordinate block path**
   The current `block_704_gray64` prototype checks all 11x11 source pixels per
   64x64 output cell and uses production-like pixel coordinates. This fixed the
   big line-edge error. On tiny no-bonus checks it can be exact or nearly exact.

4. **Current blocker**
   Bonus sprites and exact draw semantics are not fully implemented in the GPU
   prototype. The prototype uses simpler luma/circle logic for bonuses rather
   than the production sprite stamps and alpha-style composition. That is now
   the main visible parity gap.

## Quantified Gap

All rows below compare GPU output to the current CPU gray64 production
reference. H100 and L4 have the same parity behavior because they run the same
JAX code; hardware only changes speed.

| shape | bonuses | mismatch | max abs diff | mean abs diff | read |
| --- | ---: | ---: | ---: | ---: | --- |
| B1, trail 64 | 0 | 0 / 4096 pixels, `0.00%` | 0 | 0.0000 | exact tiny no-bonus oracle |
| B2, trail 256 | 0 | 0 / 8192 pixels, `0.00%` | 0 | 0.0000 | exact tiny no-bonus oracle |
| B2, trail 500 | 0 | 5 / 8192 pixels, `0.061%` | 14 | 0.0034 | tiny edge mismatch |
| B1, trail 64 | 8 | 51 / 4096 pixels, `1.25%` | 90 | 0.402 | bonus-sprite gap |
| B32, trail 512 | 8 | 123 / 8192 checked pixels, `1.50%` | 94 | 0.418 | bonus gap plus synthetic larger row |
| B64, trail 512 | 8 | 123 / 8192 checked pixels, `1.50%` | 93 | 0.379 | same code, same gap |
| B32, trail 1000 | 8 | 134 / 8192 checked pixels, `1.64%` | 98 | 0.520 | longer synthetic trail row |

Plain read:

- Trail/head rendering is close. Small no-bonus checks are exact; longer
  no-bonus checks show tiny edge mismatches.
- Bonus rendering is the real current mismatch. The mismatch touches roughly
  `1.2%` to `1.6%` of checked gray64 pixels in the synthetic bonus rows.
- The largest per-pixel differences are large (`90-98`) because a missing or
  differently-shaped sprite pixel is bright against a dark background.
- The mean error is small (`0.38-0.52` gray levels) because the mismatches are
  sparse.

## Meaning

I would not call the current GPU renderer production-equivalent yet. The gap is
not random numeric noise; it is a real semantic shortcut around bonus sprite
rendering and some exact draw-order details.

I also would not call it obviously catastrophic. The mismatched pixels are sparse
in the checked rows, and the current training evidence says approximate visual
surfaces can learn similarly when the important objects remain distinguishable.
Coach should decide the learning tolerance; Optimizer should keep quantifying
the visual delta and speed delta.

## Speed Context

The H100 made the full 704-style GPU prototype much more interesting:

| hardware | shape | device render | host->device | device->host | end-to-end | frames/sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L4/T4 | B64, trail 512 | 568.7ms | 3.2ms | 0.4ms | 573.3ms | 111.6 |
| H100 | B64, trail 512 | 57.5ms | 3.8ms | 0.5ms | 61.8ms | 1035.2 |
| L4/T4 | B32, trail 1000 | 552.1ms | 3.8ms | 0.5ms | 556.6ms | 57.5 |
| H100 | B32, trail 1000 | 61.3ms | 3.4ms | 0.3ms | 64.9ms | 493.0 |

The GPU lane is therefore real research again, especially on H100. The work is
not finished because the benchmark is still synthetic and not wired into the
stock LightZero observation path.

## Next Checks

- Add bonus sprite parity to the GPU prototype.
- Verify real env rows, not only synthetic source-state rows.
- Record mismatch location categories: trail edge, head, bonus sprite, or
  background.
- Scale-check end-to-end training, not only render: C64/C96/C128, sim8/sim16,
  L4 versus H100, with enough warmup for GPU utilization to be meaningful.
- Improve GPU utilization sampling. The current short stock profiles often
  report `0%` max util, which is probably a sampling artifact, not proof that
  the GPU is idle.
