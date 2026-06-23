# Subagent Web Research: GPU 2D Rendering For Batched Observations

Date: 2026-05-21

Status: research note only. Do not touch trainer defaults or live runs.

## Question

What is the practical GPU rendering shape for CurvyTron-style batched 2D sprite
or symbol observations, especially for `[B, P, 4, 64, 64]` RL inputs?

Short answer: do not build a general 2D renderer unless the goal changes back
to browser/artifact fidelity. For 64x64 learned observations, the practical path
is a compact, batched tensor renderer that directly writes policy-space pixels
or small masks. Sprite libraries and differentiable rasterizers are useful
references, but most of their machinery is overkill unless we need arbitrary
rotation/scaling, subpixel alpha, or exact source-resolution browser parity.

## Current Local Context

Existing notes already say the current production policy target is
`browser_lines + simple_symbols` / direct learned observations, not original
browser sprites. The newer direct GPU surface also moved the local wall away
from dense source-resolution rendering:

- `gpu_render_next_phase.md`: H100 B64 `direct_gray64` surface median around
  `0.0339s`, device render around `0.00973s`, versus dense
  `block_704_gray64` surface around `0.144s`.
- `subagent_gpu_architecture_research.md`: C512 real full-loop-ish rows are
  bounded by zero-observation at roughly `1.25x` renderer-only headroom.
- `gpu_sprite_render_research_2026-05-13.md`: original sprites require exact
  source-placement, clipping, alpha/RGB rounding, luma, and downsample behavior;
  naive 64x64 center sampling failed parity. That is a separate browser-fidelity
  project.

So this memo treats "sprites" broadly: tiny icons, heads, trails, masks, and
bonus symbols in a learned observation. It does not recommend reopening exact
browser-sprite parity unless someone explicitly asks for artifact fidelity.

## Option Read

### 1. Direct tensor stamping in JAX or PyTorch

This is the boring, likely-correct path for 64x64 observations.

Shape:

```text
compact state [B, P, ...]
-> compute visible trail/head/symbol pixel coordinates
-> write uint8/bool/float channels directly into [B, P, C, 64, 64]
-> keep tensor on device when possible
```

In JAX, `vmap` is the right outer shape for batching: JAX describes it as a
vectorizing map over argument axes. `jax.image.resize` can do nearest, linear,
area, and other resampling methods if a resize/downsample stage remains. JAX
scatter is available, but its docs warn that multiple updates to the same index
may be applied in any order, so overlap semantics need care if draw order
matters.

In PyTorch, direct writes, `scatter_`/`index_put_`, mask composition, and small
convolutions/pooling are enough for direct policy-space renderers. The main
question is deterministic overlap order. If the learned surface only needs
"heads overwrite symbols overwrite trails", build explicit masks and compose
channels in priority order instead of relying on racing scatter updates.

Practical fit:

- Best default for `simple_symbols`, heads, trails, occupancy, side lanes, and
  compact auxiliary channels.
- Works with the current trainer's PyTorch center of gravity if observations
  are handed to the model in PyTorch anyway.
- JAX is attractive if the env/search path moves that way, but a JAX renderer
  bolted to PyTorch training can add framework-transfer complexity.

Risks:

- Scatter semantics can be nondeterministic or surprising under overlap.
- Many tiny framework ops can become launch-bound; fuse once profiles show it.
- Exact draw order must be represented as tensor composition, not incidental
  write order.

### 2. `grid_sample`, `affine_grid`, and Kornia warps

This is the "sprite as transformed texture" path.

PyTorch `grid_sample` takes an input and a flow-field grid and samples 4D/5D
inputs; for 2D it maps `(N, C, H_in, W_in)` plus `(N, H_out, W_out, 2)` to
`(N, C, H_out, W_out)`. `affine_grid` generates batched 2D/3D sampling grids
from affine matrices and is meant to be paired with `grid_sample`.

Kornia wraps the same family of operations as image geometry transforms. Its
`warp_affine` accepts `(B, C, H, W)` tensors and `(B, 2, 3)` affine matrices,
with nearest or bilinear interpolation, and returns a warped `(B, C, H, W)`
tensor.

Practical fit:

- Good if symbols are actual bitmap sprites that must rotate, scale, or
  subpixel-translate.
- Good for quick experiments because it stays inside PyTorch and avoids writing
  custom kernels.
- Reasonable for a small number of sprites per row if the implementation first
  batches all sprites, warps them, then composites.

Likely overkill for current 64x64 policy observations:

- CurvyTron symbols are tiny and can be represented directly as masks or
  low-resolution stamps.
- Building a full output grid per sprite can cost more than drawing the stamp.
- Alpha blending and overlap order still need explicit composition after warp.

Recommendation:

- Use this only as a prototype lane for "rotated/scaled sprite mask" questions.
- Do not route every trail segment or static symbol through `grid_sample`.
- If used, batch all sprites as `[B * sprites, C, h, w]` and composite in a
  single priority pass.

### 3. OpenGL-style sprite batching

This is the classic game-rendering answer: put sprites in a texture atlas,
upload per-sprite transforms/UVs, draw many instanced quads in one call. OpenGL
instancing exposes `gl_InstanceID`; `glDrawArraysInstanced` behaves like
repeating `glDrawArrays` for `primcount` instances. For a game renderer, this is
the right mental model.

For RL tensors, it is usually the wrong integration boundary.

Practical fit:

- Excellent if the output is a display framebuffer or browser/artifact render.
- Useful as a conceptual model: texture atlas, one draw call, per-instance
  sprite records, GPU-side composition.

Likely overkill here:

- The training loop wants tensors, not an on-screen framebuffer.
- Reading pixels back from graphics to CUDA/PyTorch can introduce sync and
  interop complexity.
- CUDA/OpenGL interop exists, and NVIDIA documents mapping graphics resources
  into CUDA address space, but that becomes a platform-specific rendering stack
  to maintain.

Recommendation:

- Do not introduce OpenGL for the current policy observation path.
- Keep the idea of "atlas plus per-instance records" if original browser
  sprites reopen, but implement the policy surface as tensors first.

### 4. `nvdiffrast`

`nvdiffrast` is a high-performance differentiable rasterization library with
GPU-accelerated rasterization, interpolation, texturing, and antialiasing
primitives. It supports minibatches and is useful when the problem is really
triangle rasterization plus differentiable geometry.

Practical fit:

- Good for differentiable 3D/2D mesh-style rendering.
- Good if we need triangle coverage, texture sampling, antialiasing, and
  gradients through geometry.

Likely overkill here:

- CurvyTron policy observations are 64x64 raster masks, not mesh geometry.
- We do not need gradients through sprite positions for RL observation
  generation.
- The API asks us to represent sprites as triangles, clip-space vertices,
  textures, and compositing passes. That is a lot of ceremony for an icon stamp.

Recommendation:

- Do not use `nvdiffrast` for the near-term optimizer path.
- Keep it as a reference if someone asks for differentiable, antialiased,
  exact-ish vector/sprite rasterization research.

### 5. Custom CUDA, Triton, or JAX Pallas kernels

This is the highest-ceiling renderer path and the one most likely to win if the
observation wall survives full-loop profiling.

The kernel shape for direct 64x64 observations can be simple:

```text
one block/tile per row-player-channel region, or one row-player per block
read compact state/trail spans/symbol records
write contiguous [C, 64, 64] output
compose priority explicitly
optionally emit uint8 and normalize later
```

CUDA performance guidance points at memory behavior first: coalesce global
memory accesses, avoid redundant transfers, and use shared memory when it
reduces repeated global reads. For 64x64, the whole output per player is tiny,
so launch overhead, framework boundary overhead, and row packing may matter as
much as raw pixel throughput.

Practical fit:

- Best if the renderer remains a measured wall after stack/pack/readback and
  full-loop scalar boundaries are addressed.
- Best if many framework ops are visible in profiles.
- Best if active trail records can be compact and output writes are contiguous.

Risks:

- Easy to win microbenchmarks and lose the full loop by adding synchronization,
  copies, dtype conversion, or reset-order complexity.
- More correctness surface: terminal `final_observation`, autoreset, both-seat
  perspective, symbol overwrite, and RND latest-frame extraction all need gates.
- CUDA-only work narrows portability. Triton or Pallas may reduce maintenance
  if the surrounding loop is already PyTorch or JAX respectively.

Recommendation:

- Treat custom kernels as phase two, not the first research move.
- Write one only after a full-loop profile says observation is still a top wall
  and after the pure tensor renderer has proven semantics.

## What Is Probably Practical

For the next CurvyTron optimizer wave:

1. Keep the direct `64x64` learned-observation renderer as the primary lane.
2. Represent sprites/symbols as tiny masks or parametric shapes, not
   source-resolution browser sprites.
3. Compose priority with explicit masks:

```text
base/trails -> bonus/simple symbols -> heads -> terminal/side overlays if any
```

4. Keep output compact. Prefer `uint8`/bool channels inside render and convert
   near model input only if required.
5. Batch across row and player axes. The useful unit is probably `[B * P]`,
   with a stable scatter back to `[B, P, C, 64, 64]`.
6. Profile `render`, `stack/pack`, `device_to_host`, `host_to_device`, RND, and
   policy/search separately. Do not call a renderer microbenchmark a trainer
   speedup.

Simple architecture sketch:

```text
Vector manager / actors
  -> compact row-player records
  -> device direct64 renderer
       - trail occupancy/masks
       - simple symbol masks
       - head masks
       - explicit priority compose
  -> uint8 or float policy tensor
  -> batched policy/search/RND consumer
```

Near-term implementation bias:

- If staying PyTorch trainer-first: PyTorch tensor renderer, then Triton/CUDA
  only for the hottest fused step.
- If moving env/search to JAX: JAX `jit` + `vmap`, with Pallas only if XLA
  scatter/mask composition is not enough.
- Avoid OpenGL/nvdiffrast unless the goal becomes visual fidelity or
  differentiable rendering.

## What Is Likely Overkill

- Full OpenGL renderer plus CUDA interop for policy observations.
- `nvdiffrast` triangle rasterization for 64x64 icon masks.
- Per-sprite `grid_sample` for simple symbols that can be direct masks.
- Dense 704px browser canvas if the policy only consumes direct 64x64
  semantics.
- Exact original sprite alpha/RGB/luma/downsample parity for the learned
  observation lane.
- Any renderer work that does not remove host scalarization, stack packing,
  readback, or policy/search stalls in the full loop.

## Main Risks To Track

- **Overlap determinism:** scatter-like APIs may not preserve draw order when
  multiple updates target one pixel. Use masks and priority composition.
- **Launch overhead:** many small tensor ops can hide behind a nice high-level
  implementation. Profile operator count and synchronization.
- **Boundary tax:** a fast GPU render can still lose if observations bounce
  through CPU, pickle, float32 expansion, or scalar LightZero timesteps.
- **Semantic drift:** direct 64x64 is a learned observation surface, not browser
  parity. Keep that label honest in docs and metadata.
- **RND coupling:** RND can change the hot path and may require latest-frame
  extraction from `[B, P, T, C, H, W]`-like shapes. Keep it metered separately.

## Recommendation

Use a two-rung plan:

1. **Practical now:** direct batched tensor renderer for `[B * P, C, 64, 64]`,
   explicit priority masks, compact output dtype, no OpenGL/nvdiffrast.
2. **Only if profiling demands it:** fuse the proven tensor renderer into one
   custom CUDA/Triton/Pallas kernel focused on contiguous writes and compact
   row-player records.

The research-y sprite stack is tempting, but for this observation size it is
mostly theater. The interesting architecture problem is not "can a GPU draw a
sprite?" It is "can the full RL loop keep enough rows batched and resident that
drawing the sprite matters?"

## Sources

- JAX `vmap` docs: https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html
- JAX `image.resize` docs: https://docs.jax.dev/en/latest/_autosummary/jax.image.resize.html
- JAX `lax.scatter` docs: https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html
- PyTorch `grid_sample` docs: https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.grid_sample.html
- PyTorch `affine_grid` docs: https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.affine_grid.html
- Kornia geometry transform docs: https://kornia.readthedocs.io/en/latest/geometry.transform.html
- `nvdiffrast` docs: https://nvlabs.github.io/nvdiffrast/
- OpenGL `glDrawArraysInstanced` reference: https://docs.gl/gl4/glDrawArraysInstanced
- NVIDIA CUDA graphics interop docs: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html
- NVIDIA CUDA C++ best practices guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
