# Latest Rendering / Observation Audit - 2026-05-22

Scope: read the newest CurvyTron rendering, observation, and compact-loop docs/code around
`docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20`,
`src/curvyzero/env/*visual*`,
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`, and
`src/curvyzero/training/source_state_hybrid_observation_profile.py`.

Guardrails: this is a read-only rendering/env audit of code and docs. I did not edit source files,
start training, attach to live runs, or change defaults. The only workspace write is this audit note.

## Executive read

The latest rendering work is real, but it is mostly profile-only plumbing around a compact visual
observation boundary. CurvyTron now has a persistent JAX/GPU renderer, a device-only renderer output
path, a resident JAX observation stack experiment, fast compact-state adaptation from production
visual-trail arrays, native actor-side compact/render-state buffers, no-copy compact root options,
and compact replay-index proof machinery.

The important correction is that "GPU rendering" is not the same thing as "the env loop is on GPU."
Current repeated closed-loop profiles still do CPU game/env mechanics, CPU/public packaging, CPU
production-to-compact or compact-delta construction, host/device synchronization, and host-visible
search/action/replay handoffs. In the latest compact MCTX loop, search is no longer the obvious wall;
the dominant region is the env/observation boundary around actor render-state writes, observation
stack ownership, root-batch materialization, and compact replay/validation glue.

The current speed story should stay profile-scoped. Recent docs show useful gains from no-copy root
observation and the fast visual compact adapter, but the best same-denominator rows do not justify a
10x training claim. The most useful next work is lower-copy/resident observation ownership with
parity checks, not more renderer-kernel polish in isolation.

## Latest rendering work that exists

### CPU oracle and exact CPU cache reference

The older, trusted observation surface is still CPU/host oriented:

- `SourceStateBatchedObservationProfileFacade` owns a `VectorMultiplayerEnv`, host stacks, and raw
  `uint8` frames.
- `CpuOracleBatchedObservationRenderer` loops row/player views and calls the source-state gray64
  renderer.
- `SourceStateCanvasGray64DirtyRenderCache` is an exact append-only CPU dirty-block reference for
  two-player source-state gray64 rendering. It starts from a full rendered baseline and updates
  changed 11x11 source blocks. It is useful as an exactness and invalidation reference, not the
  current speed lane.

Relevant files:

- `src/curvyzero/training/source_state_batched_observation_profile.py`
- `src/curvyzero/env/vector_visual_observation.py`

### Dynamic JAX batched renderer

The dynamic GPU renderer path renders all requested rows/views by converting production state to a
compact render state, copying that state to JAX/device, drawing on GPU, and reading frames back to
host unless a device-only caller is used.

The important stages are visible in
`_render_candidate_frames_from_production_state`:

1. CPU production state to benchmark/compact render state.
2. CPU owner-ordered trail packing.
3. Host-to-device copy.
4. JAX device render and explicit readiness block.
5. Device-to-host `np.asarray(...)` readback.
6. View-major to row-major CPU reshape.

This renderer proves useful batched rendering headroom, but by itself it still has a CPU state-prep
and host-visible frame boundary.

Relevant file:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`

### Persistent JAX policy framebuffer renderer

The newest renderer lane is `_PersistentJaxPolicyFramebufferRenderer`, exposed under the
profile-only backend name `jax_gpu_persistent_policy_framebuffer_profile`. It keeps a persistent
trail layer on device and updates it from compact deltas, then composes bonuses/heads/player views
into `[B, 2, 1, 64, 64]` `uint8` frames.

What this does on GPU:

- Persistent layer reset/update via a JIT update function.
- Segment drawing into a persistent policy framebuffer.
- Bonus/head/view composition via a JIT compose function.
- Optional device-only return through `last_output_device`.

What it still does on CPU/host:

- Builds compact render state from production state.
- Computes delta segments and reset masks.
- Copies compact/delta state to device.
- Often reads frames back to host unless `device_only=True`.

The renderer is designed to fail closed for the profile backend; docs explicitly warn against hidden
CPU fallback or default/training/tournament/checkpoint changes.

Relevant file:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`

### Fast visual compact-state adapter

The latest production-to-compact improvement is
`_persistent_visual_compact_state_from_production_fast`. It consumes production visual-trail arrays
directly, slices to live trail prefixes, does minimal dtype conversion, and zeros inactive slots past
the cursor. This is the work that recent notes credit for reducing production-to-compact time from
roughly `0.37-0.52s` to about `0.054-0.057s` in matched compact-loop rows.

This is a good direction because it attacks the real boundary around render-state preparation, not
just the final GPU draw kernel.

Relevant file:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`

### Native actor compact/render-state buffers

`HybridBatchedObservationProfileManager` can run a profile-only native-buffer path through
`InProcessHybridCurvyTronActor.step_into`. The actor still calls the CPU vector env, but writes
reward/done/action mask/identity sidecars and persistent GPU render-state keys directly into parent
NumPy arrays.

This removes some payload/object churn, but it is not a GPU env step. It is a lower-copy CPU actor
handoff.

Relevant file:

- `src/curvyzero/training/source_state_hybrid_observation_profile.py`

### Device-only output and resident JAX stack experiments

The persistent renderer supports `request.device_only=True` and records its latest device output in
`last_output_device`. The MCTX synthetic benchmark can then maintain a resident JAX FIFO stack from
that latest frame instead of forcing a full frame readback and re-upload.

The current resident path is valid profile plumbing, but the latest matched rows did not yet make it
the speed recommendation. Older docs show:

- Host stack + replay rows: about `20.7k` active roots/sec.
- Host stack, no replay rows: about `17.0k` active roots/sec.
- Resident GPU stack + replay: about `16.2k` active roots/sec.
- Resident GPU stack, no replay: about `17.9k` active roots/sec.

After the no-copy root observation and fast visual compact adapter patches, docs recommend retesting
resident stack because the bottleneck moved inward.

Relevant files:

- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`

### Compact root and replay-index contracts

The profile loop now has compact root and replay-index sidecars that keep identity, legal-action,
reward/done, policy-env/player, search result, and target-row data in array-shaped contracts. This is
the right shape for a future compact/native training boundary, but the current implementation is
still a profile harness and validation surface, not the stock trainer.

Recent docs show replay-index rows are cheap enough that they should not be the primary target right
now, roughly `0.3%` wall in one matched comparison.

Relevant files:

- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`

## Current boundary: CPU mechanics vs GPU rendering vs host handoff

### Still CPU game mechanics

These pieces remain CPU/NumPy/Python in the audited paths:

- `VectorMultiplayerEnv.step(...)`.
- `VectorMultiplayerEnv._advance_runtime_for_public_step(...)`.
- `VectorRuntime.step_many(...)` / `_step_many_kernel(...)`.
- Movement, visual-trail append, body-trail append, wall checks, body collision scans, bonus catch,
  terminal score, round/tick bookkeeping.
- Public observation and info packaging through `_observe_array`, `_action_mask`, `_public_info`,
  and `_batch`.
- In-process actor loops in `HybridBatchedObservationProfileManager.step(...)`.
- Autoreset/final-observation bookkeeping.

The recent timer split matters here: actual vector runtime/physics is not currently the whole
`env_step_sec`. In late compact-loop rows, the runtime leaf is much smaller than the inclusive
env/observation bucket. CPU mechanics still exist, but the bigger observed wall is around the data
handoff and observation pipeline.

Relevant files:

- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`

### Actually GPU rendering

These pieces are actually rendered/computed on GPU in the profile renderer/search lanes:

- JAX dynamic batched frame rendering.
- JAX persistent trail-layer update.
- JAX persistent frame composition into player-view gray64 frames.
- Optional JAX resident stack update for compact visual observations.
- MCTX synthetic search in the profile benchmark.
- Torch model inference in LightZero/direct CTree profiling when CUDA is enabled.

The caveat is that GPU render/search kernels are surrounded by explicit copies and synchronizations
for measurement and compatibility.

Relevant files:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`

### Still host handoff / CPU compatibility surface

These remain host-visible boundaries or CPU compatibility costs:

- Production-state to compact render-state construction.
- Owner-ordered trail packing and live-slot selection.
- Persistent delta segment construction.
- Host-to-device copies for render state, deltas, masks, and sometimes full observation stacks.
- `block_until_ready()` timing fences in profile code.
- Device-to-host frame readback in host-stack mode.
- Host stack shift/update in `HybridBatchedObservationProfileManager._update_observation(...)`.
- `CompactRootBatchV1` construction and optional observation copy.
- Action/action-weight/root-value readback from search.
- LightZero public output decoding when using stock policy surfaces.
- Direct CTree root prepare/listify and recurrent output `.detach().cpu().numpy()` per simulation.
- Compact replay-index row construction and validation.

This is why the current wall reads as "closed compact observation/search boundary" rather than
"renderer kernel is slow."

## What the important timers mean

The top-level closed-loop timers in `mctx_synthetic_benchmark.py` are wall-clock buckets around the
profile loop. Many nested timers inside `next_step_timings_sec` are diagnostic leaves or inclusive
subspans. They should not be summed as if every field were exclusive.

### Top-level compact-loop buckets

| Timer | Meaning |
| --- | --- |
| `env_step_sec` | Inclusive wall from just before `compact_visual_manager_for_replay.step(loop_joint_action)` through the manager step and, in resident mode, the resident stack update. It is not just physics/runtime. |
| `root_build_sec` | Time to build `CompactRootBatchV1`, including sidecar assembly, validation, and optional observation copy. It is partly a profile artifact when resident/device observations are the real search input. |
| `h2d_sec` | Host-to-device boundary for observations/masks, or resident-stack handle plus mask transfer in resident mode. |
| `search_sec` | MCTX/JAX profile search wall, explicitly synchronized by reading/blocking search outputs. |
| `d2h_sec` | Search result readback: selected actions, action weights, root values, and related compact outputs. |
| `replay_index_sec` | Compact replay-index row construction from search result and step sidecars. Recent docs say this is not a major wall. |
| `unlabeled_residual_sec` | Harness overhead not assigned to the named buckets. |

### `env_step_sec` is an inclusive boundary bucket

`env_step_sec` includes:

- In-process actor stepping.
- CPU `VectorMultiplayerEnv.step(...)`.
- CPU vector runtime and reward/post-bookkeeping.
- Actor compact buffer writes.
- Actor render-state writes.
- Observation rendering/update through `_update_observation(...)`.
- Final-observation/autoreset handling.
- Compact sidecar/batch assembly and probes.
- In resident mode, the separate JAX FIFO stack update after manager step.

`env_step_sec` excludes:

- `root_build_sec`.
- Top-level search-input `h2d_sec`.
- `search_sec`.
- Search-output `d2h_sec`.
- `replay_index_sec`.

So when a profile says env is 70-80% of loop wall, that does not mean the CPU physics kernel alone is
70-80%. It means the whole action-to-next-root observation boundary is dominant.

### Actor/env timers

| Timer | Meaning |
| --- | --- |
| `actor_step_wall_sec` | Wall around the profile manager's sequential in-process actor loop. The actors are not parallel workers in this harness. |
| `actor_step_sec` | Sum of actor-side measured step/autoreset work; because actors are sequential in-process, compare with wall carefully. |
| `actor_env_step_sec` | Time inside actor `env.step(...)`, plus actor-local handling around it. |
| `actor_env_runtime_sec` | The vector runtime/physics advancement leaf inside `VectorMultiplayerEnv.step(...)`. This is closer to actual game mechanics. |
| `actor_env_public_prepare_sec` | CPU public-step setup before runtime. |
| `actor_env_reward_sec` | Reward calculation leaf. |
| `actor_env_public_info_sec` | CPU info dict/public metadata assembly. |
| `actor_env_batch_pack_sec` | CPU batch/timestep array packaging. |
| `actor_autoreset_sec` | Done-row autoreset handling. |
| `actor_compact_write_sec` | Parent compact reward/done/mask/identity sidecar writes in native-buffer mode. |
| `actor_render_state_write_sec` | Parent render-state buffer writes for persistent GPU renderer input. This has been a real wall in recent profiles. |

### Observation/render timers

| Timer | Meaning |
| --- | --- |
| `observation_sec` | Wall around `_update_observation(...)` plus reset observation handling. In host mode, this includes host stack shift/latest-frame writes; in device-only mode, it still includes render request/telemetry work. |
| `renderer_stack_update_sec` | Currently a duplicate/inclusive label for the observation update wall, not an extra exclusive span. |
| `renderer_render_sec` | Inclusive renderer telemetry span. For persistent renderer it includes compact-state prep, delta pack, H2D, persistent update, compose/render, and D2H if enabled. |
| `renderer_production_to_compact_sec` | CPU conversion from production env state to compact render fields. Recent fast visual adapter work targeted this. |
| `renderer_owner_ordered_pack_sec` | CPU owner-ordered trail packing in dynamic renderer paths. |
| `renderer_persistent_delta_pack_sec` | CPU delta/reset segment construction for persistent framebuffer updates. |
| `renderer_host_to_device_sec` | H2D copy of compact/delta/render state into JAX. |
| `renderer_persistent_update_sec` | JAX/GPU update of the persistent trail layer. |
| `renderer_device_render_sec` | JAX/GPU frame composition/draw. |
| `renderer_device_to_host_sec` | D2H frame readback. This should be zero in true device-only persistent mode. |
| `stack_shift_sec` | Host FIFO stack shift, usually `stack[:, :, :-1] = stack[:, :, 1:]`. |
| `stack_latest_update_sec` | Host latest-frame write into the last stack plane. |
| `resident_stack_update_sec` | Separate JAX FIFO stack update from `last_output_device`, measured outside manager `_update_observation` but inside top-level `env_step_sec` in the compact loop. |

### LightZero/direct-CTree timers

The LightZero/direct timers are synchronized profiling instrumentation, not invisible production
work. They deliberately call CUDA synchronization so slices can be attributed.

Important meanings:

- `model_initial_inference_sec` / `model_recurrent_inference_sec`: Torch model work, usually CUDA
  when configured.
- `root_prepare_sec`: CPU root value/policy extraction, legal-action list preparation, and CTree
  root setup.
- `model_output_d2h_sec`: tensor-to-CPU conversion for values/rewards/policies required by CTree.
- `ctree_traverse_sec` / `ctree_backprop_sec`: CPU CTree traversal/backprop.
- `public_output_assembly_sec`: public LightZero-style output conversion; recent direct-array paths
  reduce this but do not make CTree GPU-native.
- `leaf_latent_d2h_sec`: removed or reduced by `direct_ctree_gpu_latent`, but CPU CTree values and
  policy lists still remain.

The direct CTree GPU-latent path keeps latent tensors on device longer, but the tree itself remains
CPU C++/Cython plus Python/list-compatible data. It is a useful incremental profile lane, not a full
GPU MCTS architecture.

## Recent profile interpretation

Docs in the current folder describe a sequence of falsifiers and improvements:

- Stock LightZero plus `compute=gpu-*` moves model/search/learner pieces, not CurvyTron rendering.
- Scalar `jax_gpu` env/render mode is not the speed path because it still handles one env at a time
  and copies results back.
- Direct CTree/GPU-latent gives only about `1.28x-1.31x` matched full-loop improvements, not a
  `5-10x` architecture shift.
- Raw MCTX/JAX search-only throughput can be very high, but closed-loop active roots/sec collapses
  once the env/observation boundary is included.
- Observation-refresh-off rows show a real ceiling, around `1.8x-2.35x` in the cited compact-loop
  cases, but not a 10x-only-from-renderer story.
- Recent no-copy root observation plus fast visual compact adapter patches moved production-to-compact
  and root-build costs down and improved a refresh-on compact-loop row by about `1.29x`.
- After those patches, the likely wall moved inward toward actor render-state writes, observation
  stack ownership, root-batch materialization, and remaining compact handoff costs.

Representative late-row interpretation from the docs:

- A native actor buffer row around `15.9k` active roots/sec still had about `73%` of wall under
  `env_step_sec`, but that bucket included render-state/observation work.
- Observation/stack and production-to-compact were historically large enough to dominate over search.
- Replay-index rows were cheap enough not to be the main target.
- Host-stack versus resident-stack matched rows did not yet prove resident as a win, but those rows
  should be repeated after no-copy/fast-adapter changes.

## Safe optimization opportunities

These are safe because they stay in profile-only code, preserve stock training defaults, and can be
validated against existing host/oracle paths.

### 1. Retest resident GPU stack after the latest no-copy/fast-adapter patches

The old resident-stack rows did not win, but they were measured before the latest root-copy and
production-to-compact reductions. Repeat host-stack and resident-stack rows with:

- Same batch/player/sim/loop settings.
- Same active-root denominator.
- Replay on/off rows clearly separated.
- Host/resident frame parity probes.
- Terminal/autoreset/final-observation parity checks.

Success criterion: total active roots/sec improves, not just `d2h_sec` or `h2d_sec` moving elsewhere.

### 2. Let persistent renderer consume already-compact render state

The highest-probability next lever is to avoid rebuilding production-to-compact/delta inputs when the
actor already wrote the needed render-state arrays. The profile manager already has native
render-state buffers for the persistent backend; the next safe step is to make the renderer hot path
consume those buffers with sampled equivalence checks against the current adapter.

This attacks:

- `actor_render_state_write_sec`.
- `renderer_production_to_compact_sec`.
- `renderer_persistent_delta_pack_sec`.
- Redundant dtype/shape conversion.

Validation risks:

- Row/player ordering.
- Cursor regressions on reset/autoreset.
- Final observation rows.
- Partial-row request behavior.
- Visual-trail live-prefix correctness.

### 3. Keep lowering root observation materialization in resident mode

With resident stack as the real search input, building/copying a host observation inside
`CompactRootBatchV1` is mostly validation and compatibility. The `copy_observation=False` lane should
stay in profile-only experiments with sampled validation rather than full hot-loop copies.

Watch for:

- Replay-index consumers that still assume host observation availability.
- Tests that require exact host stack snapshots.
- Hidden full-stack `np.asarray` calls.

### 4. Separate measurement from synchronization side effects

Many profile timers intentionally synchronize JAX/Torch. That is correct for attribution, but a
future performance row should separate:

- Same algorithm with full profiling fences.
- Same algorithm with minimal fences and only total wall measured.

This avoids optimizing a bucket that only got large because measurement forced a block. The success
metric should remain total same-denominator closed-loop wall/active roots/sec.

### 5. Split `env_step_sec` further before changing mechanics

The current read says CPU runtime is not the whole wall. More splits are safer than mechanics changes
until the remaining top leaves are known. Good splits:

- Actor render-state write by field group.
- Compact sidecar write by field group.
- Production-to-compact fast adapter leaves.
- Persistent delta pack row loop versus dtype/array allocation.
- Host stack shift versus latest write versus reset/final-observation patching.
- Public info/batch-pack when scalar materialization is disabled.

This is low risk because it only improves attribution, and it prevents chasing the wrong CPU kernel.

### 6. Use mechanics-skipping canaries only as profile falsifiers

No-death or collision-skip variants can price the remaining CPU mechanics ceiling, especially around
body collision scans and source-frame decisions. They should remain explicit profile canaries, not
training claims, because they change game semantics.

Useful canary questions:

- How much wall remains if body collision is skipped in no-death profile mode?
- How much wall remains if public packaging is minimized in a compact-only profile?
- Does source-frame decision multiplication dominate in a specific row?

### 7. Do compact replay fast paths only if fresh timers justify them

Compact replay-index rows are the right architecture, but recent matched rows say they are not hot.
A trust-but-verified fast builder can be useful later, but it should not displace observation
ownership work unless `replay_index_sec` climbs in the same-denominator loop.

## Things not to optimize or claim yet

Avoid these as primary directions for the current wall:

- Do not flip training defaults to experimental renderer/profile modes.
- Do not touch live training runs to validate these paths.
- Do not present MCTX/toy-model search-only rows as stock MuZero training throughput.
- Do not treat scalar `jax_gpu` env mode as the main speed path.
- Do not spend more time on renderer-kernel-only polish without closed-loop same-denominator rows.
- Do not treat direct CTree GPU-latent as full GPU MCTS; the tree/list/value handoff remains CPU.
- Do not make replay-index rows the main target unless new rows show them as hot.
- Do not use exact browser-pixel or neutral/tie rendering parity as blockers for profile falsifiers
  unless the row is explicitly about production visual fidelity.

## Recommended next audit/benchmark order

1. Re-run the latest compact loop host-stack versus resident-stack comparison after no-copy root
   observation and fast visual compact-state patches.
2. Add or inspect finer splits for actor render-state writes and persistent compact/delta prep.
3. Prototype renderer consumption of already-written native compact render-state buffers in
   profile-only code.
4. Keep `copy_observation=False`/sampled-validation rows separate from full validation rows.
5. Use one semantics-changing mechanics canary only to price the remaining CPU runtime ceiling.

The north-star architecture is still compact state ownership end to end:

```text
CurvyTron compact state/action buffers
-> lower-copy CPU or native env step
-> compact render-state / resident observation stack
-> device or array-native search/model boundary
-> compact replay/target rows
-> stock LightZero adapters only at validation/compatibility edges
```

That is different from "make the renderer faster." The renderer is already fast enough in isolation
to reveal the real problem: the loop still keeps crossing host, object, stack, and compatibility
boundaries around every step.
