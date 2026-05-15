# Faithful Browser-Lines Speed Options

Date: 2026-05-15

Scope: research note for making the current policy observation target faster
without changing the trusted surface:

```text
browser_lines + simple_symbols -> controlled-player [4,64,64] stack
```

CPU `cpu_oracle` remains the semantic oracle. The JAX
`block_704_gray64` lab renderer now has exact parity on the checked
adversarial fixture rows, but the owner-priority fix made the batched renderer
slower because it carries a high-resolution priority buffer.

## Core Read

The current exact GPU block renderer avoids materializing full RGB, but it still
does approximately:

```text
B * trail_slots * 64 * 64 * 11 * 11
```

coverage tests, plus an owner-priority compare/update for every subpixel. That
is the wrong long-term shape. Exact browser-line observations can be faster
only if the renderer stops testing every trail slot against every target block,
or if the training boundary batches enough rows/views to amortize transfer and
launch overhead.

The most promising semantic trick is to avoid the priority buffer entirely.
CPU browser-lines draw order is deterministic: invalid owners first, then valid
owners descending, with each owner path ordered by slot/body number. If the
compact GPU state is already sorted into this draw order, a single overwrite
buffer is enough; later draws are exactly the owner-priority winner. This keeps
gray64 parity while removing the large per-subpixel priority image.

## Ranked Options

1. **Owner-ordered compact JAX renderer, no priority buffer.**
   Sort compact trail records per row into CPU browser draw order before device
   transfer: invalid owners, then owners descending, and stable slot/body order
   within owner. Render in that order and overwrite luma. Keep exact 11x11
   block sampling, simple symbols, heads-after-bonuses, and controlled-player
   palette.

   Expected speedup: `1.3x-2.0x` over current exact `block_704_gray64` at the
   same bucket, mostly from dropping the int priority tensor and compare/write
   traffic. Larger if sorting also enables tighter active-prefix buckets.

   Risk: medium-low. The fragile parts are cursor masking, active holes,
   invalid owners, radius discontinuities, and `break_before`. All are already
   represented in the adversarial fixture.

2. **Tile-sparse exact segment renderer.**
   Convert each trail point into same-owner segments and caps, compute target
   64x64 dirty block bounding boxes, and evaluate the 11x11 samples only for
   intersecting blocks. This can be implemented as Triton, CuPy RawKernel,
   Pallas, Numba CUDA, or C++/CUDA. The output should either process events in
   CPU draw order or write packed `(priority, luma)` values and reduce by max.

   Expected speedup: `3x-20x` over current exact full-block scanning on sparse
   rows; potentially more on early/medium trajectories. Worst case approaches
   current cost when almost every segment intersects almost every block.

   Risk: medium-high. Variable-length event lists, equal-priority tie behavior,
   clipped edges, and reset/cursor rows are where parity will break.

3. **Batched collector or vector-env observation boundary.**
   Keep many env states in one parent process, compact them, render one batch
   for all active rows and both player views, update stacks, then hand the
   batch to policy/search. Avoid scalar per-env JAX calls and avoid device-host
   round trips if policy is already on GPU.

   Expected speedup: `2x-8x` versus the scalar `jax_gpu` trainer canary; likely
   `1.2x-2x` full-loop at first, depending on MCTS/replay/learner fractions.

   Risk: medium-high integration risk. Reset/final-observation, stack FIFO,
   legal-action ordering, and subprocess CUDA behavior are bigger hazards than
   raster math.

4. **GPU dirty/incremental trail layers.**
   Maintain persistent per-owner trail layers or 64x64 dirty blocks on device,
   update only newly appended segments, then redraw bonuses/heads and
   redownsample dirty blocks. This mirrors the current CPU dirty cache idea.

   Expected speedup: `2x-6x` render speed on long append-only rollouts; smaller
   on short episodes and frequent resets.

   Risk: high for a first GPU backend because cache invalidation, palette
   changes, bonus overlap, cursor regression, and reset rows must be perfect.

5. **Full 704x704 GPU image plus CUDA/OpenCV resize.**
   Render a full source image, then area-downsample on GPU. OpenCV CUDA has
   `GpuMat` and CUDA resize/remap primitives, so downsample is easy after an
   image exists.

   Expected speedup: uncertain; likely worse than direct/tile-sparse for the
   current policy surface unless the full image is also needed for artifacts.

   Risk: medium. It recreates the large memory traffic that `block_704_gray64`
   was designed to avoid, and OpenCV line/circle rasterization is not the CPU
   oracle.

6. **Direct 64x64 center sampling or approximate analytic antialiasing.**
   Very fast, but not the same observation. Keep only as an ablation or policy
   robustness experiment, not a replacement for `cpu_oracle`.

   Expected speedup: `10x-100x` render speed.

   Risk: high semantic risk. Thin trail placement, line width, bonus symbol
   identity, and overlap order can change the training signal.

## Library Read

- JAX is still the fastest path to reuse the existing benchmark harness.
  `lax.scan`/`fori_loop` keep fixed-shape loops compilable, while Pallas could
  express a sparse tile kernel later, but Pallas is still experimental.
- Triton is a good candidate for tile-sparse kernels: it is Python-shaped but
  still close to CUDA memory/layout control.
- CuPy RawKernel is the most direct Python bridge to custom CUDA C without a
  full extension build. It is attractive for a toy event/tile kernel.
- PyTorch is attractive only at the integration boundary. A tensor/scatter
  prototype can use the same GPU as policy, but event generation and exact draw
  order are awkward with high-level ops alone.
- Numba CUDA is good for quick kernel experiments, but it adds another runtime
  and still requires careful host/device array management.
- OpenCV CUDA is useful for resize/remap once a GPU image exists; it should not
  be trusted for exact browser-line raster parity.
- Texture-style approaches fit sprite/full-image lanes better than the current
  simple-symbol policy lane. Texture cache helps 2D locality, but texture reads
  are not coherent with same-kernel writes, so read-after-write designs need
  separate kernels or global-memory staging.

## Tiny Top-Two Experiments

### Experiment 1: owner-ordered no-priority JAX

Add a benchmark-only render surface:

```text
block_704_gray64_ordered_overwrite
```

Build compact state in CPU draw order and render with one luma image, no
priority tensor. Compare against `block_704_gray64` and CPU oracle on:

- adversarial fixture, all controlled players;
- B64/S256 and B64/S1024 real-env rows;
- one long-horizon row with cursor wrap/stale tail.

Pass criterion: exact byte parity to CPU oracle and at least `1.25x` faster
than current exact priority-buffer renderer at B64/S1024 H100.

Kill criterion: any mismatch in owner crossing, invalid-owner, avatar-color,
or `break_before` rows.

### Experiment 2: tile-sparse event renderer

Prototype outside production code with either Triton or CuPy RawKernel:

```text
inputs: compact sorted segments/caps, bonus/head arrays
output: [B,1,64,64] uint8
work: one tile/program per (batch,row-block,col-block), loop only candidate
      segments whose bbox intersects the 11x11 source block
```

Start with B=4 adversarial rows and B=64 synthetic/real rows at S256. Candidate
lists can be built on CPU for the toy; the first question is renderer math, not
the perfect bin builder.

Pass criterion: exact byte parity on the adversarial fixture and `3x` faster
than current exact `block_704_gray64` at B64/S256.

Kill criterion: event-list construction plus render is slower than the ordered
JAX renderer, or parity needs tolerances.

## Recommendation

Do the owner-ordered no-priority JAX experiment first. It is the cheapest
semantic-preserving speedup and directly attacks the regression introduced by
exact owner-priority composition. In parallel, keep the tile-sparse renderer as
the bigger win path. Do not spend the next cycle on OpenCV resize, original
browser sprites, or scalar per-env JAX training integration.
