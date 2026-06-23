# Subagent Architecture Design Matrix, 2026-05-22

Status: research/design exploration. No source code, trainer defaults, live
runs, checkpoints, evals, GIFs, tournaments, or Modal state were touched.

Scope: broad architecture options for making the full
collect/search/replay-shaped loop faster. This is deliberately wider than
local shaving around the current direct CTree hook.

## Current Read

The current profile-only fast lane is:

```text
H100 compact visual MCTX
with persistent GPU policy framebuffer
and compact replay/search contracts.
```

The important bottleneck has moved. Raw draw is tiny enough that another
renderer-only pass is unlikely to be the main answer. The dominant cost in the
closed compact denominator is the state/observation/search-input handoff:

```text
selected action
-> CurvyTron state step
-> actor render-state write / production-to-compact conversion
-> persistent renderer / stack update
-> compact root batch / mask transfer
-> MCTX or CTree search
-> action readback / compact replay rows
```

The trusted production trainer still enters stock LightZero `train_muzero`
through scalar CurvyTron env rows. The current train-facing
`direct_ctree_gpu_latent + output-fast` profile signal is real, but it stays in
the `~1.28x-1.31x` full-loop class because it keeps the old LightZero
collector/tree/replay object topology. The bigger 5-10x lane needs a different
owner for compact state, search inputs, search outputs, replay, and RND.

## Speedup Classes

| Class | Meaning |
| --- | --- |
| Tactical | `1.05x-1.30x` on a matched compact or train-profile row. Useful, not a new architecture. |
| Boundary | `1.30x-2.00x` by deleting one hot object/copy/sync boundary. |
| Structural | `2x-5x` by keeping compact batches alive across multiple stages. |
| Rewrite | `5x-10x+` only plausible when env/search/replay ownership changes together. |

## Matrix

| # | Design | What It Removes | Expected Speedup Class | Risk | Validation Gate | Touches Production Trainer? |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Borrowed render state canary | Parent render-state copy for single-actor, no-terminal profile rows. | Tactical to Boundary: `1.10x-1.35x` compact closed-loop if `actor_render_state_write_sec` collapses. | Low-medium. Terminal/autoreset final-frame ordering is sharp. | Copied vs borrowed observation/stack/mask/reward parity; borrowed mode fails closed on terminal rows; at least `10%` matched H100 improvement. | No. Profile-only flag only. |
| 2 | Live-prefix and delta render-state ownership | Rebuilding full trail/render state every step; copying dead/old trail capacity; redundant persistent framebuffer deltas. | Boundary: `1.2x-1.8x` compact loop if render-state handoff is still dominant. | Medium. Trail cursor, break markers, reset, and final observation can drift. | Delta/live-prefix parity over seeded rows, terminal reset fixtures, and timing proof that production-to-compact plus delta pack mostly vanish. | No first. Could feed trainer only after compact path promotion. |
| 3 | Resident compact state owner | Production env state as source of truth for hot visual/search path; repeated production-to-compact conversion. | Structural prerequisite: `1.5x-3x` compact loop, possibly more when paired with compact replay/search. | Medium-high. Changes ownership of state facts, not just storage. | Scalar oracle parity for positions, alive/done, rewards, masks, render state, final observations, and replay sidecars across resets. | No in prototype. Yes if promoted as optimized trainer env. |
| 4 | Persistent GPU stack/root batch | Host `[B,P,4,64,64]` stack update, root observation copy, repeated obs H2D, and hot-loop stack validation copies. | Boundary to Structural: `1.2x-2x` compact MCTX loop, larger only if obs H2D/readback is the live wall. | Medium. Easy to move synchronization into search timing without reducing total wall. | Resident stack sampled parity, `obs_h2d_bytes == 0` for search input, legal masks correct, total closed-loop roots/sec improves `1.2x+`. | No. Profile-only until trainer consumes compact roots. |
| 5 | Batched CTree/search boundary | Python list CTree APIs, per-simulation CPU reward/value/policy payloads, per-root output dicts. | Tactical for flat-A3 CTree in full loop; Structural if replaced by compiled/device-resident search. | Medium for flat-A3, high for replacement search. Semantics and root-noise parity are hard. | Forced masks exact, illegal visit mass zero, clear-preference exact, root-noise statistical pass, same-denominator full-loop row `>=1.5x` for tactical or `>=3x` for replacement. | No first. Train-facing direct hook remains profile-gated; replacement would require trainer adapter. |
| 6 | Actor/search service split | Synchronous scalar collect/search interleave; GPU underfilled by one collector batch; per-env action dict boundary. | Structural to Rewrite: `3x-10x` if many compact actor batches feed one saturated search/model owner. | High. Changes collection cadence, staleness, replay age, and failure modes. | Producer/consumer mock saturates search service; compact chunks preserve row/player/reward/done/visit/value identity; learner-shaped sampler survives without scalar fallback. | Not initially. Yes for real training architecture. |
| 7 | Vectorized CPU alternative | Scalar env wrappers, per-row Python stepping, object payload merge. Uses SoA NumPy/Numba/CPU vector slabs instead. | Boundary to Structural: `1.5x-4x` on env/update side if GPU search needs CPU supply. | Medium. Less radical than GPU/JAX, but collision/render details can become branchy. | `step_many` parity against scalar fixtures; env/update rows/sec beats current manager by `2x+`; GPU search remains fed without queue starvation. | No if sidecar. Yes if replacing production env manager. |
| 8 | C++/Rust extension | Python loops in env step, collision/raster, ring buffers, replay packing, maybe CTree search loop shell. | Structural for env/obs hot paths: `2x-5x` on those buckets; full-loop depends on remaining search/replay. | High. Packaging, CI, Modal build, debugging, and semantic freeze risk. | Native `step_many` or compact replay writer exact against golden fixtures; benchmark beats NumPy/Numba by `2x+`; no Python objects in hot loop. | No for isolated extension probe. Yes if adopted as trainer backend. |
| 9 | JAX full env/search prototype | Framework boundary between env/render/search; PyTorch/LightZero CTree CPU shell; host stack/root bounce. | Rewrite: `5x-10x+` possible in search-heavy lanes if env, obs, model, search, and replay become JAX-shaped. | Very high. Requires alternate learner/checkpoint/replay story or weight conversion. | Fixed-shape JIT loop compiles once, legal masks exact, MCTX output maps to `CompactSearchResultV1`, interop copy does not erase win, same-scale sim16/sim32 beats direct by `3x+`. | No. Scratch prototype only until a separate trainer lane exists. |
| 10 | PufferLib-style vector env/replay slab | Repeated allocation/return of env payload dataclasses, scalar timestep materialization, redundant obs/reward/mask copies. | Structural: `2x-5x` if compact env/search/replay sidecar stays contiguous; supports Rewrite lanes. | Medium-high. It is a system contract, not a drop-in trainer. | Static buffer owner exposes obs/action/reward/done/mask/final/replay sidecars; compact consumer runs search/RND/replay without scalar fallback; `>=3x` over current direct profile denominator. | No first. Yes if it becomes the main optimized collection path. |

## Design Cards

### 1. Borrowed Render State Canary

Shape:

```text
single in-process actor owns env.state
-> renderer borrows actor env.state directly
-> compact sidecars still written into parent arrays
-> no scalar LightZero materialization
```

This is the smallest useful ownership experiment. It removes the parent
`_native_render_state` write for no-terminal single-actor profile rows. It does
not solve compact state ownership, but it prices one currently visible bucket.

Validation gate:

- copied and borrowed modes produce identical observation, resident stack,
  action mask, reward/done, row/player ids, and compact replay-index rows;
- terminal/autoreset rows raise a clear profile-only error until a pre-reset
  snapshot protocol exists;
- `actor_render_state_write_sec` falls below noise and matched H100 closed-loop
  throughput improves at least `10%`.

Production trainer touch: none. Keep this as a profile-only falsifier.

### 2. Live-Prefix And Delta Render-State Ownership

Shape:

```text
actor updates live trail prefix/cursor/delta facts
-> persistent framebuffer consumes only live deltas
-> old/dead trail capacity stays resident and untouched
```

This goes beyond borrowing. The renderer should not receive a freshly rebuilt
view of all production trail arrays every step. It should receive stable
resident state plus tiny deltas: changed head positions, newly written trail
segments, reset rows, live-prefix lengths, and bonus changes.

What it removes:

- full render-state row copies;
- dead/old trail capacity scans;
- repeated production-to-compact packing for unchanged rows;
- excessive persistent framebuffer delta work.

Validation gate:

- trail cursor wrap, `break_before`, alive/dead rows, reset rows, and terminal
  final frames match copied render state;
- delta telemetry proves changed slots scale with live writes, not total trail
  capacity;
- closed compact loop improves `1.2x+` on rows where renderer handoff is the
  top bucket.

Production trainer touch: no first. This belongs in the compact profile lane.

### 3. Resident Compact State Owner

Shape:

```text
compact CurvyTron state is the hot source of truth
-> env step updates compact state in place
-> renderer/stack/root/search/replay borrow from that state
-> stock production objects exist only for validation/debug
```

This design attacks the current ownership mismatch directly. Instead of
converting production state to compact renderer/search state every step, the hot
loop keeps the compact structure alive across steps.

What it removes:

- production-to-compact conversion;
- actor payload merge for state already known to the compact owner;
- repeated row/player identity reconstruction;
- separate state copies for render, root batch, replay, and RND.

Validation gate:

- seeded scalar parity for position, heading, alive, collision, reward, done,
  action mask, final observation, reset, and row/player perspective;
- compact replay index rows preserve the same record identity as the scalar
  adapter;
- no hidden `BaseEnvTimestep` or per-root dict allocation in the hot path.

Production trainer touch: no in the falsifier. Yes if this replaces the stock
env manager in an optimized trainer.

### 4. Persistent GPU Stack And Root Batch

Shape:

```text
persistent GPU renderer writes latest frame on device
-> device FIFO maintains [B,P,4,64,64]
-> CompactRootBatchV1 points at resident roots
-> MCTX consumes resident obs and persistent masks
-> sampled validation copies only every N steps
```

The point is not merely "use GPU memory." The point is to stop bouncing:

```text
GPU render -> host stack -> root batch copy -> device_put -> search
```

What it removes:

- hot host stack updates when a device consumer exists;
- root observation copies;
- repeated obs H2D for MCTX;
- some explicit `block_until_ready` timing artifacts if total wall improves.

Validation gate:

- sampled resident vs host FIFO parity, including reset rows and newest-frame
  order;
- `obs_h2d_bytes` for search input goes to zero while mask transfer is reported
  separately;
- total closed-loop roots/sec, not just bucket timing, improves `1.2x+`.

Production trainer touch: none until a trainer consumes compact root batches.

### 5. Batched CTree/Search Boundary

There are two subdesigns with very different ceilings.

Conservative bridge:

```text
fixed A=3 CTree APIs accept arrays
reward[N], value[N], policy[N,3], masks[N,3]
```

This removes Python nested-list payloads but keeps CPU CTree and Python
simulation control. Flat-A3 already proved no-model boundary speedups, while
matched full-loop rows did not show a trainer-level win. Keep it as a
diagnostic bridge, not the 10x plan.

Larger replacement:

```text
compact roots on device
-> fixed-shape tree arrays
-> batched recurrent calls
-> compact selected_action/visit_policy/root_value
```

This can be compiled Torch, Triton/CUDA, JAX/MCTX, or deeper C++/Cython. The
requirement is that reward/value/policy do not return to Python lists every
simulation.

Validation gate:

- forced single-legal and clear-preference cases exact;
- illegal visit mass zero;
- root-noise collect mode statistical match;
- no-noise eval mode deterministic enough for documented tolerance;
- same-denominator profile: `>=1.5x` for tactical CTree changes, `>=3x` for a
  replacement search service.

Production trainer touch: profile-only first. Any replacement must pass replay
and target parity before trainer promotion.

### 6. Actor/Search Service Split

Shape:

```text
compact actor batches
-> ready queue of obs/mask/reward/done/state ids
-> central batched model/search service
-> compact actions/visit/value
-> compact replay writer
-> learner-shaped sampler
```

This is the broad systems rewrite. It borrows the MiniZero/KataGo idea of many
positions feeding batched inference/search, but keeps CurvyTron's row/player
compact contracts.

What it removes:

- synchronous scalar collect/search interleave;
- per-env action dicts;
- GPU underfill from one small facade call at a time;
- public LightZero collector output fanout in the optimized lane.

Validation gate:

- producer/consumer mock shows the search service stays saturated;
- chunk metadata records policy version, env row, player, action mask, reward,
  done, final observation, visit policy, and root value;
- compact replay sampler can build learner-shaped tensors without scalar
  timesteps;
- bounded staleness and replay age are explicit.

Production trainer touch: not for the mock. Yes for any real training adoption.

### 7. Vectorized CPU Alternative

Shape:

```text
SoA CPU state arrays
actions[B,P]
step_many(...)
observe_many(...)
reward/done/mask buffers
optional Numba kernel for branch-heavy loops
```

This is the pragmatic alternative to an all-GPU or all-JAX environment. If the
GPU search service is fast, a highly vectorized CPU env may be enough to keep
it fed while avoiding a risky GPU simulator.

What it removes:

- one Python env object per scalar LightZero row;
- per-step dict/list wrappers;
- actor payload dataclasses for the hot data;
- repeated row/player order reconstruction.

Validation gate:

- scalar oracle parity on wall, self, opponent, head-head, terminal, and reset
  fixtures;
- per-stage timing for movement, collision, observation, masks, replay packing;
- `step_many` plus observation beats current profile manager by `2x+` at
  B512/B1024 and does not starve GPU search.

Production trainer touch: no if kept as sidecar. Yes if it replaces the stock
env manager.

### 8. C++/Rust Extension

Shape:

```text
native step_many / observe_many / pack_replay_many
with pointers into compact arrays
optional pybind/cffi/pyo3 boundary
```

This should not mean "rewrite everything in C++." LightZero CTree already has
C++ kernels. The useful native extension is the one that removes the Python
boundary that profiling has proven hot: env step, collision/raster, replay
packing, or search-loop shell.

What it removes:

- Python control loops in proven-hot kernels;
- NumPy scatter/gather overhead if it dominates;
- Python object allocation around replay/target rows;
- possibly the CTree Python simulation shell if a deeper native search owner is
  built.

Validation gate:

- native arrays exact against golden scalar/vector fixtures;
- cross-platform Modal build is reproducible;
- native backend beats best Python/NumPy/Numba by `2x+` on the targeted bucket;
- the full compact loop improves, not just the kernel microbench.

Production trainer touch: no for isolated probes. Yes if adopted as a backend.

### 9. JAX Full Env/Search Prototype

Shape:

```text
JAX state[B,2,...]
-> jitted step/render/stack
-> MCTX search over roots[B*2]
-> selected actions scatter back to state[B,2]
-> compact replay arrays
```

This is the cleanest way to test the all-device premise. It should be treated
as a separate prototype, not a LightZero patch. A PyTorch model inside a jitted
MCTX recurrent function would recreate the bad boundary.

What it removes:

- PyTorch/LightZero/CTree framework boundary;
- CPU tree state and Python list payloads;
- host stack/root transfer;
- scalar env manager and timestep objects.

Validation gate:

- fixed-shape JIT compiles once and steady-state timing excludes compile time;
- legal masks and padded/dead roots are correct;
- output maps to `CompactSearchResultV1`;
- interop copy is reported if any PyTorch/NumPy bridge remains;
- same-scale sim16/sim32 beats current direct CTree by `3x+` after warmup.

Production trainer touch: none. It needs a separate replay/learner/checkpoint
story before it can be trainer advice.

### 10. PufferLib-Style Vector Env/Replay Slab

Shape:

```text
one static slab owns:
  obs_uint8[B,P,4,64,64]
  action[B,P]
  reward[B,P]
  done[B]
  legal_mask[B,P,3]
  final_observation and autoreset masks
  replay/search sidecars
workers write into fixed slices
consumer reads contiguous ready batches
```

The lesson is not to use PufferLib as the learner. The lesson is to copy the
buffer contract: static memory, direct writes into owned slices, contiguous
transfer, optional pinned/device mirrors, and scalar compatibility only at
validation edges.

What it removes:

- actor payload allocation and merge;
- scalar `BaseEnvTimestep` materialization in the optimized hot path;
- redundant observation/reward/mask copies;
- anonymous arrays that lose replay and terminal identity.

Validation gate:

- slab fields cover everything currently carried by `HybridCompactBatch` and
  `CompactReplayIndexRowsV1`;
- compact search/RND/replay consumer uses the slab without scalar fallback;
- terminal/final/autoreset rows stay exact;
- closed compact search/replay loop beats current direct profile denominator
  by `3x+`, or this is not the 5-10x path.

Production trainer touch: no for the slab prototype. Yes if the optimized
collector is promoted.

## Recommended Order

1. Keep the borrowed render-state canary as the smallest current-wall price.
2. In parallel, design the live-prefix/delta owner because that is the next
   coherent state-handoff fix if borrowing wins.
3. Build resident GPU stack/root-batch rows with sampled validation copies.
4. Use the Puffer-style slab and resident compact state as the main structural
   design, so search/RND/replay consume one compact owner.
5. Keep flat-A3 CTree as a boundary probe, not the main speed plan.
6. Use actor/search service and JAX full env/search as radical reference lanes,
   gated by same-denominator `3x+` wins and replay/learner-shaped costs.

## References Read

- `README.md`
- `current_hot_path_bottleneck_map_20260522.md`
- `gpu_mcts_current_flow_explainer_20260522.md`
- `subagent_state_ownership_big_moves_critique_20260522.md`
- `subagent_next_state_ownership_patch_20260522.md`
- `device_stack_handoff_plan_20260521.md`
- `puffer_style_contiguous_buffer_attach_audit_20260522.md`
- `native_vector_buffer_architecture_plan_20260522.md`
- `array_native_ctree_next_design_20260522.md`
- `subagent_search_replay_architecture_plan_20260522.md`
- `subagent_compiled_search_architecture_20260522.md`
- `subagent_external_patterns_20260522_late.md`
- `docs/research/performance_vectorization.md`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/training/compact_policy_row_bridge.py`
