# Post-Search Hot Path Audit

Date: 2026-05-22

Role: parallel code/dataflow auditor.

Scope:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/training/exploration_bonus.py`
- `src/curvyzero/training/compact_policy_row_bridge.py`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/*`

No live Coach training runs were touched.

## Plain Read

The current docs are directionally right: once search gets fast, the next wall is
the closed compact loop, not another isolated renderer microbench.

The compact path is useful because it stops some scalar LightZero object fanout,
but it is still not a device-resident training loop. It still does a lot of
work as Python objects, NumPy arrays, copied sidecars, validation adapters,
RND lists, and host-visible renderer/search outputs.

So the next question is not:

```text
Can MCTX/JAX search be fast?
```

We already have a strong yes for the search-boundary probe.

The next question is:

```text
Can one repeated loop stay compact and cheap?

search -> selected joint action -> env step -> observation/stack update
-> replay/RND/target edge -> next search
```

That is the denominator that matters now.

## Still CPU, Python, Or Object-Shaped

### Hybrid Actor And Env Loop

`HybridBatchedObservationProfileManager.step` still loops through Python actor
objects and row partitions. In the normal path it builds per-actor payload
objects, then merges them into compact arrays. In the `native_actor_buffer`
path it writes directly into `self._compact`, but that path currently rejects
renderer-backed observations.

Important lines:

- `source_state_hybrid_observation_profile.py:506-540`: native actor buffer
  exists, but only for zero-observation rows.
- `source_state_hybrid_observation_profile.py:540-562`: non-native path loops
  over actors and then calls `_merge_payloads`.
- `source_state_hybrid_observation_profile.py:746-758`: each returned step
  copies policy ids, reward, done, action masks, and the compact payload.

Plain read: even in the compact profile lane, the manager is still a Python
orchestrator around NumPy arrays. That is okay for proof tooling, but it is a
wall after search gets fast.

### Observation Stack

The stack is still host-side NumPy. Every step shifts the whole stack and writes
the latest frame.

Important lines:

- `source_state_hybrid_observation_profile.py:779-817`: stack shift, render
  call, `np.asarray(result.frames)`, and latest-frame write all happen through
  host-visible arrays.
- `source_state_hybrid_observation_profile.py:819-864`: autoreset does another
  render path and host stack update for reset rows.

Plain read: this is no longer obviously the biggest wall by itself, but it is
part of the closed-loop wall. If the renderer/search are fast, shifting and
copying `[B,2,4,64,64]` on the host every tick becomes visible again.

### Persistent GPU Renderer Boundary

The persistent renderer is GPU-backed, but the setup around it is still host
work. The delta builder uses Python loops over rows and slots, then the renderer
returns policy frames that are copied back into host arrays for the stack.

Important lines:

- `source_state_batched_observation_boundary_profile.py:3032-3120`: persistent
  delta construction loops in Python over batch rows and trail slots.
- `source_state_batched_observation_boundary_profile.py:3239-3283`: JAX compose
  draws heads/bonuses on device.
- `source_state_hybrid_observation_profile.py:800-814`: renderer output is
  observed as NumPy and written into the host stack.

Plain read: the GPU render kernel is not the whole render cost. The surrounding
host delta construction and host stack write still matter.

### Search Boundary

The direct CTree/GPU-latent path keeps latent states on GPU, but the CTree API
still wants CPU/list-shaped root prep and per-simulation backprop inputs.

Important lines:

- `source_state_batched_observation_boundary_profile.py:5513-5527`: root values
  and policy logits are copied from GPU to CPU before root prep.
- `source_state_batched_observation_boundary_profile.py:5535-5545`: policy
  logits, legal actions, and noises are turned into Python lists/root objects.
- `source_state_batched_observation_boundary_profile.py:5575-5593`: telemetry
  explicitly tracks GPU-latent D2H and listification buckets.
- `source_state_batched_observation_boundary_profile.py:5606-5724`: output still
  comes through `roots.get_distributions()`, `roots.get_values()`, NumPy arrays,
  Python loops for masked rows, and copied compact search arrays.

Plain read: this is why direct CTree gave a real but modest full-loop win. It
removed one bad copy path, but it did not remove the CPU/list CTree contract.

### Compact Replay Bridge

`CompactReplayIndexRowsV1` is the right idea because it avoids copying
observations in the collection hot path. But building and validating it still
does many NumPy conversions and copies. Materializing full target rows is still
a per-row Python dict loop and must stay out of the hot path.

Important lines:

- `compact_policy_row_bridge.py:124-240`: `build_compact_root_batch_v1` validates
  and copies observation, masks, ids, rewards, done flags, and final-observation
  sidecars.
- `compact_policy_row_bridge.py:414-576`:
  `build_compact_replay_index_rows_v1_from_search_result` avoids observation
  copies, but still validates/copies many side arrays.
- `compact_policy_row_bridge.py:579-742`:
  `materialize_compact_target_rows_from_index_rows_v1` loops over rows, builds
  dicts, and copies `observation` plus `next_observation`. This is validation or
  sampler-edge code, not hot collection code.

Plain read: compact index rows are the right bridge shape, but the current
validation-heavy builder may become a wall in the repeated closed loop.

### RND

RND is still very object-shaped.

Important lines:

- `exploration_bonus.py:481-536`: scalar RND extraction converts inputs to
  NumPy/CPU and slices latest frames.
- `exploration_bonus.py:582-626`: compact RND extraction also starts by forcing
  the observation and target reward through `_to_numpy_cpu`.
- `exploration_bonus.py:849-872`: `collect_data` walks segment objects and
  extends a Python list of cloned torch tensors.
- `exploration_bonus.py:874-904`: `train_with_data` samples from that Python
  list, stacks tensors, moves them to the device, and writes metric snapshots.
- `exploration_bonus.py:906-962`: `estimate` computes on torch, then reads back
  CPU stats, normalizes inside the batch, deep-copies target rewards, and
  returns list-shaped data.

Plain read: RND is not the biggest measured bucket after hash fixes, but it is
not compatible with a resident compact loop. If search becomes much faster, RND
will be one of the next things that reintroduces host/object overhead.

## Likely Walls After Search Gets Fast

Ranked by current Amdahl risk:

1. Closed compact env/observation/replay edge.
   The MCTX rows showed huge search-boundary speed, but the one-step
   search-plus-replay denominator fell back to a few thousand roots/sec. That
   points at env step, observation update, replay-index construction, and their
   synchronization.

2. CPU/list CTree contract if we stay inside LightZero CTree.
   Direct GPU-latent search is useful, but it still converts root/recurrent
   payloads through CPU arrays and Python lists. A 5-10x move probably needs
   array-native search or a compact search service, not another tiny wrapper
   patch.

3. RND/replay/learner adapters.
   RND and target-row materialization are still list/object/CPU lanes. They may
   look small while search is slow, but they will matter if search is moved to a
   much faster service.

4. Host stack copies.
   Host stack movement is not the top wall in every current row, but it is a
   repeated full-batch copy. Once search and replay are cleaner, this becomes a
   likely next wall unless the policy/search path consumes a resident or ring
   stack.

## Most Plausible Next Code Changes

### 1. Add A Fast, Trust-But-Verified Compact Replay Index Builder

Keep the current `build_compact_replay_index_rows_v1_from_search_result` as the
strict validator. Add a second internal builder for the repeated profile loop
that assumes a previously validated `CompactRootBatchV1` shape and writes/copies
only the arrays needed for `CompactReplayIndexRowsV1`.

Plain target:

```text
strict builder:
  use at reset, debug gates, and periodic validation

fast builder:
  use inside the closed compact loop after shape/identity are already proven
```

Why this is plausible:

- It is small and local.
- It does not touch live Coach training.
- It attacks the replay-index edge that became visible after fast MCTX search.
- It keeps the strict path available as a guard.

Expected risk:

- Medium semantic risk if the fast path skips too much. Mitigation: run strict
  and fast side by side for sampled steps and assert byte/value equality.

### 2. Let `native_actor_buffer` Work With Renderer-Backed Compact Rows

Right now the direct compact actor buffer path fails closed when an observation
renderer is present. That forces renderer-backed rows through payload objects
and render-state merging.

Plain target:

```text
actor.step_into(...)
  -> writes compact env/result arrays
  -> writes or exposes render-state arrays in the same compact buffer
  -> renderer consumes the compact buffer without payload object merge
```

Why this is plausible:

- The code already has `native_actor_buffer` for zero-observation rows.
- The current rejection is explicit and narrow.
- It attacks actor payload objects, `_merge_payloads`, `_merge_render_state`,
  and extra copies before observation update.

Expected risk:

- Medium to high shape-contract risk because render state has more arrays than
  reward/done/action masks. Mitigation: start profile-only, add strict equality
  against the current payload path for a few steps, then run the closed-loop
  profile.

## Do Not Do Next

- Do not keep polishing `body_circles_fast` or old renderer modes. The active
  optimized surface is browser-lines/simple-symbols policy observations, and
  the search/replay boundary is now the larger question.
- Do not sell fresh MCTX roots/sec as full-loop speed. It is search-boundary
  speed, not closed-loop training speed.
- Do not move target-row materialization into the hot path. It is explicitly a
  validation/sampler adapter.
- Do not touch live Coach training runs for this lane.

## Recommended Next Profile

Run one small closed compact loop row with per-step buckets:

```text
B256 or B512
sim16
persistent policy framebuffer
closed-loop steps >= 8 after warmup
record:
  search_sec
  env_step_sec
  render/delta/stack_sec
  compact root build sec
  compact replay-index sec
  RND off first
```

Then rerun with the fast replay-index builder or native renderer-backed actor
buffer change, whichever is implemented first.

The keep/kill rule should be simple:

```text
Keep the patch only if it moves closed-loop active roots/sec, not just an
isolated sub-bucket.
```
