# GPU Sprite Render Research

Date: 2026-05-13

Scope: research note for moving CurvyTron bonus sprite stamping onto GPU while
preserving the trusted visual surface.

## Plain Goal

Render active bonus sprites fast enough that they are no longer visible in the
training profile. The target is not a new visual style. The target is the
current full-fidelity path: browser-style trails, browser-sprite bonuses, heads,
then BT.601 luma and 11x11 downsample.

## What The Current CPU Path Does

Current source-state sprite data:

- sprite sheet path:
  `third_party/curvytron-reference/web/images/bonus.png`;
- grid: 3 columns by 4 rows;
- bonus type names map to sprite indices in
  `SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_NAMES`;
- CPU stamps are cached by `(sprite_index, dst_size)`;
- stamp math uses nearest source sampling, alpha blend, round, clip, then writes
  back to the RGB canvas.

Draw order matters:

1. background and trail layer;
2. active bonus sprites;
3. live player heads;
4. luma/downsample to gray64.

That means a GPU path must preserve alpha blending for sprites and must still
allow heads to appear on top of bonuses.

## External Research Anchors

Known graphics pattern: put many sprites into one texture atlas or texture
array, then batch all sprite draws. NVIDIA’s texture guidance says atlases help
reduce state changes and enable larger batches. It also warns that texture
updates and host/device transfers are expensive, so static atlas data should be
uploaded once and reused.

CUDA guidance says GPU work pays off when there are many parallel data elements,
memory access is coherent, and host/device transfers are minimized. It also
says data should stay on the device across kernel calls when possible.

For this repo, the direct lesson is simple: upload the bonus sprite atlas once,
keep it on GPU, send compact per-env sprite metadata, and render/downsample in
one batched device path. A GPU path that copies a whole RGB frame back to CPU
after every tiny sprite pass is the wrong shape.

References:

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

## Best Current Shape

Best production-shaped GPU design:

- static sprite atlas lives on GPU;
- compact source-state arrays move to GPU in batches, or eventually stay there;
- one fused or staged GPU renderer draws trails, alpha-blends bonus sprites,
  draws heads, computes luma, and writes `[B, P, 1, 64, 64]`;
- readback is avoided unless the trainer requires CPU tensors.

GPU research update: the classic graphics answer is one atlas plus one batched
or instanced draw list with stable blending. For this training renderer, the
production shape is probably a CUDA/PyTorch extension or a JAX custom/Pallas
kernel only if the rest of the hot path is JAX-native. Keep the atlas,
primitive lists, 704 scratch, and final observation on device. Use premultiplied
RGBA or otherwise exactly match current source-over alpha semantics. Preserve
draw order: trails, sprites, heads. Avoid unordered sprite-parallel writes if
two translucent primitives can overlap; a per-pixel or per-tile gather path is
safer for parity.

Simplest useful prototype:

- no live trainer;
- fixed batch of source-state arrays;
- GPU atlas tensor;
- one active sprite per env row;
- compare byte output against CPU `render_source_state_canvas_gray64`;
- then add sprite-over-trail and head-over-sprite tests.

Implementation sketch for the first serious prototype:

1. Upload the 3x4 sprite atlas once.
2. Copy compact per-row draw metadata once per benchmark batch.
3. Run one kernel to produce the 704 RGB parity surface.
4. Run one kernel to apply BT.601 luma plus exact 11x11 area downsample.
5. Compare CPU/GPU output by bytes before optimizing.
6. Later fuse render and downsample when parity is boring.

Broader research update: exact direct-to-64 is possible, but not as pure luma
accumulation. Sprite alpha blending happens per RGB channel and rounds before
luma, so the exact algorithm is a block-local RGB rasterizer. For each dirty
64x64 output cell, compose only its 11x11 source tile in the exact current draw
order, then immediately compute luma and the area average. This avoids a full
704x704 materialization while preserving semantics.

That same shape works on CPU and GPU:

- CPU: persistent trail masks plus dirty 11x11 tile recomposition.
- GPU: one thread block or warp group per dirty output tile and player
  perspective, using the sprite atlas and compact primitive lists on device.

Near-exact luma-domain sprite blending remains a possible approximation lane,
but it is not the trusted full-fidelity route because RGB alpha rounding can
change the final gray64 value.

Toy confirmation:

- scratch probe: `/private/tmp/curvy_sprite_probe/sprite_block_probe.py`;
- exact per-pixel block accumulator: `0` mismatches in `250/250` placements;
- exact interval/count accumulator: `0` mismatches in `250/250` placements;
- naive 64x64 center sampling: `247/250` placements mismatched;
- luma-space blending before RGB rounding: `127/250` placements mismatched.

So the clean trick is not to sample the sprite at 64x64. The trick is to
preserve the current 704-pixel nearest-neighbor/alpha/rounding semantics while
accumulating only the source pixels that affect dirty output blocks.

## What To Avoid

- Do not claim GPU sprite replacement until the bonus sprite parity gate passes.
- Do not optimize around `body_circles_fast`; it is not the current target.
- Do not copy the full RGB canvas to and from GPU per step.
- Do not measure only one short trajectory and call it a training speedup.
- Do not launch one kernel per sprite or let Python loop over sprites on the hot
  path.
- Do not call `.cpu()`, `.numpy()`, `.item()`, or any host read inside the
  repeated render path.
- Do not rely on unordered atomics for alpha blending unless the parity tests
  prove overlap ordering is irrelevant.

## Open Questions

- Is the current CPU hot spot sprite alpha blending itself, full-frame redraw of
  all bonuses, or dirty-block invalidation around bonuses?
- Can CPU dirty-block-limited bonus redraw remove most of the cost before GPU is
  needed?
- Can LightZero consume the rendered observation while it is still on GPU in the
  trusted stock path?
