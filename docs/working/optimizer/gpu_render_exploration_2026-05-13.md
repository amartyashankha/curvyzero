# CurvyTron GPU Render Exploration

Date: 2026-05-13

Scope: isolated optimizer experiments for CurvyTron visual rendering. Do not
touch live training runs. The current trusted training lane remains stock
LightZero `source_state_fixed_opponent`.

Current target: full-fidelity `browser_lines` only. That means source-state RGB
on the 704-style canvas, BT.601 luma, then 11x11 downsample to 64x64.
`body_circles_fast` rows below are historical/control measurements, not a
current GPU optimization lane.

## Plain Goal

Find out whether GPU rendering can improve real exploration/training
throughput, not just a render microbenchmark.

The current CPU baseline is stronger now: cached `browser_lines` avoids
redrawing old trail history and gives a large long-trajectory speedup locally.
So GPU has to beat that stronger baseline or unlock a different architecture.

## Current Read

GPU is promising only if the workload is batched and the output stays close to
the policy/search path.

Update after first Modal probes: the raw GPU math is not the blocker. On an L4,
synthetic direct-64 render kernels are sub-millisecond at useful batch sizes.
The bigger risk is moving data around and preserving the real observation
contract.

Bad shape:

- one env step at a time;
- CPU state copied to GPU;
- tiny render kernel;
- rendered frame copied back to CPU;
- policy/search then consumes CPU data.

Good shape:

- many env rows at once;
- compact source-state arrays copied in bulk, or eventually kept on device;
- one or a few fused kernels;
- output is `[B, P, 4, 64, 64]` or similar policy input;
- policy/search consumes the GPU tensor without immediate CPU readback.

## First Toy Benchmark

Build a standalone Modal GPU function. It should not import LightZero and should
not write to training volumes.

Measure these pieces separately:

- CPU synthetic state generation;
- CPU -> GPU transfer for compact primitive arrays;
- device-only direct-64 splat renderer;
- optional GPU -> CPU readback;
- equivalent simple CPU renderer baseline;
- warmup versus steady-state timing.

The first renderer does not need full browser-line fidelity. It should emulate
the workload shape: batched rows, many trail/body primitives, small 64x64 output,
and per-player color/luma values.

Implemented:

- `src/curvyzero/infra/modal/curvytron_gpu_render_probe.py`
- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`

Both are isolated Modal apps. They do not use training volumes, checkpoints, or
live runs.

The first CuPy attempt failed on Modal image plumbing: NVRTC/CUDA root discovery
was not clean in the simple base image. I switched both probes to the repo's
known-good JAX CUDA image pattern, which already works for the Mctx benchmarks.

## Results So Far

These numbers are L4/T4 Modal GPU runs. They are prototype economics, not a
training-speed claim.

### Direct64 Probe

`curvytron_gpu_render_probe.py` renders synthetic circles directly into
`[B,64,64]`. It is intentionally not browser-line fidelity.

| B | primitives | device render | host->device | device->host | end-to-end with readback | read |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 64 | 0.140ms | 0.881ms | 0.161ms | 1.190ms | readback loses to CPU |
| 1 | 256 | 0.246ms | 0.934ms | 0.247ms | 1.489ms | barely useful with readback |
| 8 | 64 | 0.183ms | 0.903ms | 0.214ms | 1.348ms | batching starts to help |
| 8 | 256 | 0.214ms | 0.856ms | 0.216ms | 1.239ms | clear GPU win vs toy CPU |
| 64 | 64 | 0.135ms | 0.705ms | 0.225ms | 1.085ms | transfer dominates |
| 64 | 256 | 0.311ms | 0.792ms | 0.277ms | 1.350ms | transfer still large |

Plain read: GPU render itself is extremely fast for this tiny output. A
CPU->GPU->CPU loop is bad at `B=1`, becomes reasonable by `B=8`, and is mostly
transfer-limited by `B=64`.

### Source-State Direct Gray64 Probe

`source_state_gpu_render_benchmark.py` uses synthetic source-state arrays with
trail slots, heads, bonuses, and a `browser_lines` approximation. It now has
two surfaces:

- `direct_gray64`: samples one point per 64x64 cell. Fast economics probe.
- `block_704_gray64`: samples all 11x11 source pixels per 64x64 cell and
  averages luma. This is closer to the active 704->64 cost, but it still does
  not materialize a full RGB canvas or claim browser parity.

The `body_circles_fast` entries in these tables are retained only as controls.
Do not use them for recommendations.

| mode | B | trail slots | device render | host->device | device->host | end-to-end | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `browser_lines` | 16 | 100 | 0.366ms | 4.132ms | n/a | 4.498ms | parity checked vs prototype CPU |
| `browser_lines` | 16 | 500 | 0.469ms | 3.379ms | n/a | 3.760ms | parity checked vs prototype CPU |
| `browser_lines` | 64 | 500 | 0.981ms | 3.910ms | n/a | 5.052ms | device work still small |
| `browser_lines` | 64 | 500 | 1.100ms | 3.191ms | 0.287ms | 4.617ms | readback of output is small |
| `browser_lines` | 64 | 2000 | 4.705ms | 3.078ms | n/a | 7.797ms | long trails make render matter |
| `browser_lines` | 128 | 500 | 2.072ms | 3.630ms | n/a | 5.874ms | batching improves throughput |
| `browser_lines` | 128 | 2000 | 5.175ms | 3.473ms | n/a | 8.662ms | render and transfer both matter |
| `body_circles_fast` | 64 | 500 | 0.648ms | 4.495ms | n/a | 5.107ms | simpler kernel, same transfer tax |
| `body_circles_fast` | 128 | 500 | 0.772ms | 2.675ms | n/a | 3.454ms | very fast approximation row |

Plain read: for the direct-gray64 prototype, GPU render is already fast enough
that copying state to the GPU is often the larger bucket. For long trails
(`S=2000`), render becomes comparable to transfer. This supports two next moves:

- prototype a closer full-fidelity 704->64 renderer only if it stays batched;
- avoid per-step CPU round trips in the eventual training path.

`block_704_gray64` rows:

| mode | B | trail slots | device render | host->device | device->host | end-to-end | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `browser_lines` | 4 | 100 | 2.436ms | 3.662ms | n/a | 6.121ms | first 11x11-subpixel row |
| `browser_lines` | 16 | 100 | 6.373ms | 3.170ms | n/a | 9.561ms | render now larger than transfer |
| `browser_lines` | 16 | 100 | 6.456ms | 4.530ms | 0.392ms | 11.332ms | readback is small vs render/transfer |
| `browser_lines` | 8 | 500 | 16.502ms | 3.850ms | n/a | 20.383ms | long trail cost is real |
| `browser_lines` | 1 | 64 | 0.867ms | 2.887ms | 0.387ms | 4.200ms | no-bonus production comparison |
| `browser_lines` | 1 | 64 | 0.991ms | 3.327ms | 0.227ms | 4.864ms | bonus-sprite gap visible |
| `browser_lines` | 16 | 64 | 4.738ms | 3.336ms | n/a | 8.026ms | batched economics row |
| `browser_lines` | 2 | 256 | 3.671ms | 3.313ms | 0.258ms | 7.222ms | exact no-bonus two-row oracle |
| `browser_lines` | 16 | 256 | 23.123ms | 3.210ms | n/a | 26.333ms | long-trail render dominates copy |
| `browser_lines` | 2 | 500 | 7.327ms | 5.973ms | 0.479ms | 16.453ms | near-exact no-bonus two-row oracle |
| `browser_lines` | 16 | 500 | 46.716ms | 3.171ms | n/a | 50.021ms | naive full GPU redraw gets expensive |

Plain read: a closer 704-style cost is no longer "free", but it is not
catastrophic on L4 either. The measured cost is in milliseconds for batches,
not seconds. The likely production answer is not a naive full batch redraw for
every step; it is either dirty 11x11 block updates or a fused exact-ish renderer
that avoids rechecking old unchanged trails.

The benchmark now has a tiny production-render comparison for
`block_704_gray64`. It maps synthetic rows into production source-state keys and
compares against `render_source_state_canvas_gray64(..., trail_render_mode="browser_lines")`.
The first version sampled world-space centers and was visibly wrong on line
edges. After switching the block path to production-like 704-pixel coordinates
and fixed trail radius, the no-bonus line/head path is close but not exact:

| B | trail slots | bonuses | mismatch | max diff | mean diff | CPU reference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 0 | 0 / 4096 pixels (exact) | 0 | 0.0000 | 192ms |
| 1 | 64 | 8 | 51 / 4096 pixels (1.25%) | 90 | 0.402 | 208ms |
| 2 | 256 | 0 | 0 / 8192 pixels (exact) | 0 | 0.0000 | 216ms |
| 2 | 500 | 0 | 5 / 8192 pixels (0.061%) | 14 | 0.0034 | 332ms |

Plain read: the no-bonus line/head path can now hit byte-exact parity on this
tiny oracle. The renderer is still not a trusted replacement because the bonus
row is worse: the prototype still uses simple scalar circles while production
uses sprite stamps. At longer trail counts, naive full redraw also becomes
expensive enough that dirty/incremental rendering remains the likely production
shape.

## Decision Gates

Continue toward a real GPU renderer only if:

- device-only render is clearly faster at useful batch sizes;
- transfer plus render is still competitive when output stays on GPU;
- readback cost is measured and understood;
- kernel launch count is low;
- speed improves with larger `B` instead of only looking good in isolation.

Pause or deprioritize if:

- transfer/readback dominates at realistic batch sizes;
- launch overhead dominates because work is too tiny;
- CPU cached `browser_lines` remains faster for the same end-to-end shape;
- the prototype only wins by changing observation semantics too much.

## Full-Fidelity Next Plan

1. Feed real source-state rows into `source_state_gpu_render_benchmark.py`
   instead of only synthetic rows.
2. Tighten strict parity against CPU `browser_lines`: RGB canvas, BT.601 luma,
   and 11x11 downsample to 64x64. The first tiny oracle exists and currently
   shows small but real mismatches.
3. Measure CPU state packing separately. Host->device is already large enough
   that packing/copy shape matters.
4. Test a Torch handoff shape: output device tensor -> policy input without CPU
   readback.
5. Compare against the current CPU cached `browser_lines`, not old full-redraw
   baselines and not `body_circles_fast`.
