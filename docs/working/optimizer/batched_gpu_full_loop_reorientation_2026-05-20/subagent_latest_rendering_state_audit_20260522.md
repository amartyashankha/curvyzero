# Latest Rendering State Audit - 2026-05-22

Scope: read-only audit for the current CurvyTron rendering / observation
state. I reviewed the batched GPU optimizer docs in this folder, the rendering
docs under `docs/working/optimizer/*render*` and
`docs/working/training/*render*`, and the requested source files:

- `src/curvyzero/env/observation_surface_contract.py`
- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`

Guardrail: no source files, live runs, Modal jobs, checkpoints, tournaments, or
training state were touched. This file is the only intended output.

## Executive Read

The current trusted policy observation surface is still:

```text
browser_lines + simple_symbols -> cpu_oracle -> [4,64,64]
controlled-player perspective, 4-frame stack
```

The current fast renderer/profile lane is:

```text
jax_gpu_persistent_policy_framebuffer_profile
+ direct_gray64
+ browser-line policy-space semantics
+ simple_symbols
+ profile-only compact/hybrid/MCTX probes
```

Do not optimize old render paths by accident. `body_circles_fast`,
`fast_gray64_direct`, scalar `policy_observation_backend=jax_gpu`, and
browser-sprite parity notes are stale or diagnostic unless an experiment is
explicitly labeled as a legacy/control benchmark.

The latest rendering wins are real, but the current wall has moved from raw
device draw to the closed compact observation/search-input boundary:

```text
selected action
-> CPU env / actor step
-> render-state handoff
-> production-to-compact / delta pack
-> persistent GPU renderer
-> host or resident stack ownership
-> CompactRootBatchV1 / search input
-> search result / replay-index edge
```

Recent docs repeatedly warn that `env_step_sec` in compact MCTX rows is not
"game mechanics"; it is the whole action-to-next-observation/search-input
handoff. That is the bucket the next optimizer should keep splitting and
shrinking.

## Current Trusted Policy Observation Surface

The authoritative contract is `observation_surface_contract.py`.

Current constants:

- `POLICY_TRAIL_RENDER_MODE = "browser_lines"`
- `POLICY_BONUS_RENDER_MODE = "simple_symbols"`
- `POLICY_RENDER_SURFACE_LABEL = "browser_lines+simple_symbols"`
- `POLICY_STACK_SHAPE = (4,64,64)`
- `POLICY_RAW_DTYPE = "uint8"`
- `POLICY_MODEL_DTYPE = "float32"`
- `DEFAULT_POLICY_OBSERVATION_BACKEND = "cpu_oracle"`
- `CURRENT_RELIABLE_POLICY_OBSERVATION_BACKEND = "cpu_oracle"`
- `EXPERIMENTAL_SCALAR_POLICY_OBSERVATION_BACKEND = "jax_gpu"`
- production direction: `batched_gpu_observation_backend_not_scalar_jax_gpu`

Code reference:
`src/curvyzero/env/observation_surface_contract.py:30-68` and
`:74-122`.

Trusted surface semantics:

- render source-state trails as `browser_lines`;
- draw active bonuses as `simple_symbols`;
- draw heads after bonuses;
- use BT.601 luma;
- downsample from the 704-style source canvas to 64x64 by 11x11 area average;
- expose a controlled-player view;
- keep newest frames in a 4-plane policy stack.

Important wording: this is a policy-observation contract, not a browser-pixel
claim. CPU `cpu_oracle` is the reliable backend today. The target future is a
batched GPU backend that preserves the policy surface, not the scalar `jax_gpu`
env backend.

`compute=gpu-*` in stock LightZero means model/search/learner CUDA. It does not
mean CurvyTron observations are rendered on GPU.

## Current GPU Renderer Status

### Isolated renderer benchmark

`source_state_gpu_render_benchmark.py` is an isolated Modal benchmark, not a
trainer or live-run path. It defines:

- render mode: `browser_lines`;
- bonus modes including `simple_symbols`;
- surfaces: `direct_gray64` and `block_704_gray64`;
- state sources: synthetic, real env rollout, and adversarial fixture.

Code reference:
`src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py:29-75`.

The benchmark metadata states the split plainly:

- `direct_gray64` samples/draws directly in 64x64 policy space and is a fast
  economics probe, not trusted browser fidelity.
- `block_704_gray64` outputs 64x64 while checking high-resolution sample
  positions, closer to the CPU reference but still no materialized full RGB
  canvas.
- real env rows are benchmark-only; there is no trainer/checkpoint integration
  in this file.
- browser sprite parity is out of scope when using `simple_symbols`.

Code reference:
`src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py:230-287`.

### Dynamic JAX batched renderer

The dynamic renderer path remains:

```text
CPU production state
-> CPU compact render state
-> CPU owner-ordered trail pack
-> H2D copy
-> JAX render
-> D2H readback
-> view-major to row-major CPU conversion
-> host stack update
```

Code reference:
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2692-2758`.

This path is truly batched for render, but not resident end to end. It is still
host-owned around state prep, readback, and stack update.

### Persistent JAX policy framebuffer

The active fast profile backend is
`jax_gpu_persistent_policy_framebuffer_profile`. It is explicitly constrained
to `render_surface="direct_gray64"` and only allowed for canary/profile rows.

Code reference:
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1009-1040`.

What it does:

- keeps a persistent `[B,2,64,64]` trail layer on device;
- tracks previous cursor, owner position, owner validity, and avatar-color
  invalidation state;
- builds compact/delta inputs on host;
- copies deltas and compose state to device;
- updates the persistent device layer;
- composes bonuses/heads/player views;
- stores `last_output_device`;
- reads frames back unless `request.device_only=True`.

Code reference:
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2819-2965`.

Known file-level gap is also explicit:

```text
profile-only Modal sidecar; not wired into trainers, tournaments, eval,
checkpoints, or live runs
GPU output is read back to host before stack update
persistent GPU framebuffer backend is profile-only and policy-space direct64;
it is not browser pixel parity
```

Code reference:
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:8929-8936`.

### Hybrid observation profile

`source_state_hybrid_observation_profile.py` is a profile-only in-process
hybrid scaffold. It can use zero stacks or renderer-backed stacks. It can
materialize scalar LightZero timesteps at the edge, or feed a batched compact
consumer.

Current profile result metadata says:

- `profile_only=True`;
- `calls_train_muzero=False`;
- `stock_lightzero_integrated=False`;
- `trainer_defaults_changed=False`;
- `touches_live_runs=False`.

Code reference:
`src/curvyzero/training/source_state_hybrid_observation_profile.py:1398-1408`.

The manager still has host stack ownership by default:

- host FIFO shift at `_zero_stack[:, :, :-1] = _zero_stack[:, :, 1:]`;
- renderer request with row-major rows/players;
- `np.asarray(result.frames)`;
- host latest-frame write into the stack.

Code reference:
`src/curvyzero/training/source_state_hybrid_observation_profile.py:1039-1084`.

It now also has `borrow_single_actor_render_state` guardrails for a
single-actor, profile-only no-terminal canary:

- requires `native_actor_buffer=True`;
- requires `actor_count=1`;
- requires an observation renderer;
- requires refresh-on observation stack.

Code reference:
`src/curvyzero/training/source_state_hybrid_observation_profile.py:525-574`.

## Known Fidelity Gaps

The current trusted CPU surface is the only production-trusted policy
observation. The GPU profile surface is useful but approximate.

Current GPU/direct gaps:

- `direct_gray64` is policy-space direct64, not the 704 RGB canvas plus exact
  11x11 downsample path.
- `direct_gray64` can match its CPU-direct oracle on focused canaries, but that
  is not exact parity against the production CPU oracle.
- `block_704_gray64` is a closer renderer probe, but the benchmark still says
  no full RGB canvas parity and no browser sprite parity.
- Original browser sprite-sheet rendering is artifact/reference work, not the
  current policy target.
- Persistent renderer rows are speed/profile evidence, not browser-pixel
  evidence.
- Tolerant divergence rows must be called divergence telemetry, not exact
  parity.

High-risk contract edges that remain important:

- row/player order: renderer output can be view-major while consumers expect
  row-major `[row0p0,row0p1,row1p0,row1p1,...]`;
- controlled-player perspective and `avatar_color` changes;
- stack FIFO order and dtype normalization;
- terminal `final_observation` before autoreset;
- partial reset/autoreset rows;
- cursor regression, game clear, stale active tail slots, and prefix mutation;
- bonus-over-trail and head-over-bonus draw order;
- RND latest-frame extraction from compact/uint8/float stacks;
- fixed-opponent `to_play=-1` and active-root filtering.

Relevant audit docs:

- `subagent_gpu_observation_validation_plan_20260521.md`
- `subagent_gpu_renderer_boundary_20260521.md`
- `subagent_latest_renderer_surface_audit_20260522.md`
- `subagent_latest_rendering_audit_20260522.md`

## What Has Already Been Optimized

Rendering and observation work that should be treated as already done or
already measured:

1. CPU `browser_lines` dirty/incremental cache landed as the exact CPU
   reference direction. It reduces long-trail full redraw and keeps full render
   fallback semantics.
2. Stationary bonus dirty-block invalidation was fixed. Bonus boxes are no
   longer dirtied every step when unchanged.
3. `simple_symbols` became the current policy bonus representation. V8
   row-specific masks keep all 12 bonus identities separable.
4. The old direct-fast/body-circles CPU lane was superseded as a production
   recommendation; keep it only as old speed/control evidence.
5. Dynamic JAX renderer learned the important semantics: previous same-owner
   browser-line connectivity, break-before handling, simple-symbol bonuses,
   owner-priority composition, two-view output, and real-env state ingestion.
6. `direct_gray64` became the current drastic profile surface because it avoids
   dense 704-block draw cost.
7. Renderer-backed trainer/profile surfaces now fail closed if no explicit
   renderer is provided or the backend label mismatches.
8. Persistent policy-space GPU framebuffer was implemented as profile-only
   `jax_gpu_persistent_policy_framebuffer_profile`.
9. Persistent synthetic benchmark rows showed large renderer-side speedups
   with exact parity against the synthetic stateless target.
10. The real persistent profile path now trims render state to the live visual
    or body cursor prefix instead of copying a full `body_capacity=4096` tail.
11. The fast visual compact-state adapter cut `production_to_compact` from
    roughly `0.37-0.52s` to about `0.054-0.057s` in matched compact rows.
12. No-copy `CompactRootBatchV1` observation mode removed a root-batch copy in
    profile rows.
13. After fast visual plus no-copy root, resident GPU stack became a measured
    win in the relevant compact profile rows: about `26.6k -> 31.6k` sim16 and
    `21.2k -> 28.9k` sim32 in the cited rows.
14. Compact replay/index rows were proven cheap enough not to be the next
    target in the current denominator.
15. The profile loop now has grouped renderer telemetry: production-to-compact,
    delta pack, H2D, persistent update, D2H, stack/update, and actor
    render-state-write buckets.

Non-rendering but nearby optimizations already measured:

- `direct_ctree_gpu_latent` plus output-fast gives a real matched stock-loop
  profile gain around `1.28x-1.31x`, but it is not a 5-10x architecture.
- Flat-A3/array CTree work is useful ABI evidence but did not become the main
  full-loop path.
- CPU64 did not help the current search-boundary rows.

## Stale Or Old Code Paths

Treat these as stale unless explicitly labeled as control/legacy/fidelity work:

- `body_circles_fast`: old/generic render ablation and historical CPU fast
  surface. Not the current trusted policy surface.
- `fast_gray64_direct`: old/custom two-seat naming. Do not copy old commands;
  current equivalent profile language is `render_surface=direct_gray64` plus
  the persistent GPU profile backend and canary guards.
- `browser_sprites`: artifact/GIF/browser-fidelity lane, not the policy
  observation target. Human GIFs should still use rich browser-lines rendering,
  but policy observations use simple symbols.
- Scalar `policy_observation_backend=jax_gpu`: experimental scalar env backend;
  one row at a time, readback to NumPy, slower/failure-prone in subprocess
  stock paths. It is not the batched GPU renderer.
- Any command that says H100 or `compute=gpu-*` and then implies GPU CurvyTron
  rendering. H100 compute alone only moves LightZero compute to GPU.
- `block_704_gray64` as the primary speed lane. It remains useful for closer
  fidelity probes, but current fast profile rows use `direct_gray64`.
- Renderer-kernel-only work as the main next lane. Device draw is no longer the
  only or obvious wall in current compact-loop rows.
- More replay-index machinery in the current shape. Replay/index rows are
  already cheap in the cited denominator.
- Exact neutral/tie parity as a promotion blocker for search rows. LightZero
  CTree itself does not repeat neutral tie visits exactly stock-vs-stock.

## What The Next Optimizer Should Not Forget

1. Label the currency of every speed claim:

```text
actual training
stock train_muzero full-loop profile
profile-only boundary probe
synthetic renderer/search microbench
```

Do not compare roots/sec from a compact profile to env steps/sec from
`train_muzero` as if they were the same claim.

2. Keep the two surfaces separate:

```text
trusted production/training/tournament:
  browser_lines + simple_symbols + cpu_oracle

fast optimizer/profile:
  jax_gpu_persistent_policy_framebuffer_profile + direct_gray64
```

3. Do not call `env_step_sec` game mechanics in compact MCTX rows. It includes
render-state handoff, observation update, stack ownership, root input prep, and
reset/final-observation work.

4. Resident claims need telemetry proof. A future row labeled resident should
say whether full-frame D2H, `np.asarray(output_device)`, host stack update,
host obs H2D, `.cpu().numpy()`, or scalar timestep materialization happened in
the measured loop.

5. If optimizing the latest path, start from the current hot profile shape:

```text
native_actor_buffer=True
actor_count=1 unless testing actor topology
persistent GPU profile renderer
direct_gray64
fast visual live-prefix adapter
no-copy root observation where valid
compact replay-index proof separated from target-row materialization
```

6. The next likely renderer-side move is ownership, not another renderer
kernel:

- split actor render-state writes by field family;
- prototype borrowed/already-compact render-state consumption;
- keep or build a device-resident observation stack consumed by MCTX/search;
- validate sampled host/device stack parity;
- fail closed on terminal/autoreset until pre-reset snapshots are explicit.

7. Keep RND separate. RND latest-frame extraction, meter mode, hashing,
positive reward, and cadence are separate axes. Do not fold RND rows into a
renderer speed claim.

8. Keep stock LightZero adapters at validation/compatibility edges. The
compact path should not reintroduce scalar `BaseEnvTimestep` fanout, public
LightZero output dicts, or target-row object materialization into the hot
profile loop unless the experiment explicitly prices that edge.

9. Promotion gates remain unsatisfied for production:

- trainer/tournament defaults unchanged;
- checkpoint metadata plan not promoted for GPU profile backend;
- natural death/autoreset/final-observation surface still needs explicit
  profile gates;
- same-surface persistent-vs-stateless parity needs to stay exact;
- direct64-vs-CPU-oracle drift must stay labeled as drift until a new surface
  is promoted.

10. Do not touch live Coach runs while working this lane. Use profile-only
Modal/local harnesses and artifact-producing manifest runners.

## Recommended Next Checkpoint For The Main Optimizer

Before any new code patch, re-read these files in order:

1. `task_board.md`
2. `README.md`
3. `subagent_latest_renderer_surface_audit_20260522.md`
4. `subagent_latest_rendering_audit_20260522.md`
5. `subagent_mechanics_vs_observation_audit_20260522.md`
6. `subagent_next_state_ownership_patch_20260522.md`
7. `subagent_state_ownership_big_moves_critique_20260522.md`
8. `current_hot_path_bottleneck_map_20260522.md`

Then verify the code path still names:

```text
observation_renderer_backend=jax_gpu_persistent_policy_framebuffer_profile
render_surface=direct_gray64
render_mode=browser_lines
bonus_render_mode=simple_symbols
profile_only=True
calls_train_muzero=False for compact/hybrid profile probes
```

If those labels are missing, stop and prove what path is actually being
optimized before trusting any timing row.
