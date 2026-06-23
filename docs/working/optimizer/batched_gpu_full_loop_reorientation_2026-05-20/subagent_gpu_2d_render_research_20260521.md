# Subagent GPU 2D Render Research

Date: 2026-05-21

Scope: research brief only. No code, trainer defaults, live runs, checkpoints,
or tournament artifacts changed.

## Bottom Line

The current CurvyTron GPU renderer is paying for long-tail redraw: every frame
re-tests accumulated trail slots even though game semantics mostly append a
small number of new trail/body segments. The most practical creative direction
is therefore not a richer 2D renderer. It is a stateful render surface:
maintain a persistent trail layer, update only changed segments, then compose a
small dynamic head/symbol layer into `[B, P, 1, 64, 64]`.

That idea only matters if it survives the repo's current boundary: CPU env
state, CPU compact/pack, H2D copy, JAX render, D2H readback, CPU stack update,
then scalar LightZero timestep materialization. Existing local notes already
show that direct `64x64` rendering removed most dense device-render cost, while
C512 zero-observation still leaves a large host/manager/search floor. So the
renderer experiment must measure end-to-end-with-readback and not celebrate a
device-only toy in isolation.

## Local Context

- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py` already has
  the right current semantics: `browser_lines`, `simple_symbols`,
  `direct_gray64`, fused two-view render, owner-priority composition, and CPU
  verification rows.
- `docs/.../gpu_render_next_phase.md` says H100 B64 `direct_gray64` surface was
  about `0.0339s` versus `block_704_gray64` about `0.144s`; device render was
  about `0.00973s` versus `0.123s`. That is the big dense-render win.
- `docs/.../subagent_gpu_renderer_boundary_20260521.md` maps the remaining
  boundary: CPU compact state, H2D, GPU render, D2H, row-major conversion,
  host float32 stack, and scalar LightZero objects.
- `docs/.../subagent_gpu_system_patterns_20260521.md` gives the Amdahl warning:
  best real batched GPU C512 around `1439.84 steps/s`, zero-observation ceiling
  around `1805.22 steps/s`, so renderer-only upside in that stock-shaped path
  is roughly `1.25x`.

## Useful External Patterns

- GPU RL systems win when observation generation stays on GPU. CuLE ports Atari
  emulation to CUDA, renders frames directly on GPU, and reports up to 155M
  frames/hour on one GPU while avoiding CPU/GPU bandwidth as the bottleneck
  ([NVIDIA CuLE](https://research.nvidia.com/publication/2020-12_accelerating-reinforcement-learning-through-gpu-atari-emulation)).
  PixelBrax makes the same modern point in JAX: physics plus pure JAX renderer
  can run pixel-observation RL end-to-end on GPU and render thousands of envs
  ([PixelBrax](https://arxiv.org/abs/2502.00021)).
- CPU can still be the right answer when the loop is CPU-shaped. EnvPool's
  C++ environment engine targets parallel env execution and reports 1M Atari
  FPS and 3M MuJoCo FPS on high-end hardware
  ([EnvPool](https://arxiv.org/abs/2206.10558)). For CurvyTron, this argues
  that a native persistent CPU renderer inside actor workers may beat a GPU
  renderer that immediately copies pixels back to Python.
- Scatter must encode overlap semantics. JAX's scatter docs warn that multiple
  updates to the same index may be applied in any order
  ([JAX scatter](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html)).
  For CurvyTron, use explicit priority buffers, owner-ordered passes, or
  `amax`-style encoded priority values; do not rely on racing writes to decide
  head/bonus/trail order.
- CUDA kernels should write contiguous output. NVIDIA's CUDA guide emphasizes
  coalesced global memory access and shows worst-case strided writes wasting
  most memory traffic
  ([CUDA memory performance](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#memory-performance)).
  A renderer kernel should make consecutive threads write consecutive pixels in
  row-major `[row_player, y, x]` order where possible.
- OpenGL/EGL is viable for headless graphics, but it is probably the wrong
  training boundary. EGL can create off-screen OpenGL contexts without X, and
  FBOs plus CUDA/OpenGL interop can avoid extra copies
  ([NVIDIA EGL](https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/),
  [CUDA graphics interop](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html)).
  Instanced sprite rendering is a good game-rendering model
  ([glDrawArraysInstanced](https://docs.gl/gl3/glDrawArraysInstanced)), but
  RL still needs tensors, stack semantics, and often D2H data.
- PyTorch warps are useful for real sprites, not trail redraw. `grid_sample`
  maps 4D image tensors plus normalized sampling grids to output tensors and is
  often paired with `affine_grid`
  ([PyTorch grid_sample](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.grid_sample.html)).
  It is handy for rotated/scaled icon experiments, but too heavy for thousands
  of trail segments.
- `nvdiffrast` is excellent but over-scoped here. It supplies CUDA-accelerated
  rasterization, interpolation, texturing, and antialiasing primitives with
  minibatch support
  ([nvdiffrast](https://nvlabs.github.io/nvdiffrast/)). CurvyTron policy
  observations do not need differentiable triangle rasterization.
- Triton and JAX Pallas are plausible custom-kernel routes if profiles justify
  it. Triton is a Python-based GPU kernel language aimed at high-throughput
  custom kernels ([Triton docs](https://triton-lang.org/main/index.html));
  Pallas is JAX's experimental custom-kernel path for GPU/TPU
  ([JAX Pallas](https://docs.jax.dev/en/latest/pallas/)).

## Ranked Options For This Repo

1. **Persistent policy-space trail framebuffer, plus dynamic overlay.**
   Maintain `trail_layer[B, P, 1, 64, 64]` across steps. On each env step,
   stamp only newly emitted trail segments/body points into the persistent
   layer, clear affected rows on reset/game-clear, and compose heads/bonuses in
   a separate transient layer. This directly attacks long-tail redraw. Best
   first as a toy because it tests the core hypothesis without a full trainer
   rewrite. Main risks: terminal final-observation timing, game-clear erasure,
   overlap priority, and state residency.

2. **Active-span/tile full redraw kernel.**
   Keep stateless rendering but make work proportional to active visible spans,
   not configured trail capacity. Use active-prefix/truncated slots, tile
   culling, row-player tiles, and owner-priority composition. This fits the
   current benchmark code and is lower-risk than a persistent renderer, but it
   still redraws old trails every frame.

3. **Native CPU persistent renderer in C++/Rust.**
   If LightZero continues to demand host observations, a CPU-native persistent
   `uint8` frame per actor may be better than GPU render plus D2H. The env is
   already CPU-side, 64x64 is tiny, and actor parallelism matters. This option
   is especially attractive if profiles show H2D/D2H/stack/scalarization larger
   than device render.

4. **Central batched GPU render service.**
   Preserve CPU actor parallelism, aggregate compact render requests, render
   larger batches on GPU, return `uint8` frames. This is the right architecture
   experiment if one-process batched manager saturates. It loses if IPC plus
   copies simply wrap the same CPU-compact/H2D/render/D2H path at small batches.

5. **PyTorch or JAX tensor stamping with explicit masks.**
   Useful as a quick prototype and possibly enough for `simple_symbols`, heads,
   and short active spans. Compose `trail -> bonus -> head` with masks or
   encoded priorities. Avoid duplicate-index scatter races. Watch for many
   tiny ops becoming launch-bound; `torch.compile`, Triton, or Pallas can fuse
   later if this path proves semantics.

6. **OpenGL/EGL FBO plus instanced sprites.**
   Great if the requirement returns to browser/display fidelity or large sprite
   atlases. Poor first choice for policy tensors because it creates graphics
   context, interop, readback, and deployment complexity.

7. **`grid_sample`, Kornia, nvdiffrast, general differentiable renderers.**
   Keep as references. Use only if the observation contract demands rotated
   bitmap sprites, antialiased geometry, or differentiable rasterization.

## Minimal Toy Experiment Proposal

Goal: test whether persistence removes the long-tail redraw slope and whether
the win survives the current readback/stack boundary.

Use the existing benchmark state shape rather than a trainer:

```text
B = {64, 256, 512}
P = 2
target = 64x64
trail_capacity = {256, 512, 1024}
T = 512 synthetic steps
events = append 1 segment/player/step, occasional row reset, occasional game-clear,
         fixed bonuses, moving heads
```

Compare three variants:

- **A: stateless full redraw.** Existing `direct_gray64 + browser_lines +
  simple_symbols` semantics, redrawing all selected trail slots.
- **B: persistent GPU delta.** Device-resident `trail_layer`; each step H2D
  sends only new segment records and clear masks, stamps deltas, composes
  transient heads/symbols, then optionally D2H returns `uint8` frames.
- **C: persistent CPU delta.** Same semantics in a simple native/NumPy baseline
  first; later C++/Rust if the NumPy sketch shows a promising shape.

Metrics:

- device/update time, H2D, D2H, row-major conversion, stack write, total with
  readback;
- slope versus accumulated trail length;
- exact frame equality versus variant A at every 16th step and at reset/clear
  steps;
- output bytes and dtype path (`uint8` until model boundary).

Pass bar:

- Persistent variant's update time is approximately flat as accumulated trail
  length grows.
- It matches stateless direct frames on adversarial overlap/reset/clear cases.
- It gives at least `3x` device-side improvement at `B512/S512` and at least
  `1.3x` end-to-end-with-readback improvement. If it only wins without
  readback, it is a device-residency research result, not a current-loop fix.

Decision rule: if persistence wins device-only but loses end-to-end, stop
renderer-kernel work and move to either CPU-native actor rendering, central GPU
render service, or broader LightZero payload/stack residency work.

