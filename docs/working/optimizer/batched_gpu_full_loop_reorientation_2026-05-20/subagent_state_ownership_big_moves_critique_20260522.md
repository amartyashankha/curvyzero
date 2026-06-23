# State Ownership Big-Moves Critique, 2026-05-22

Scope: read-only architecture critique. I read the current optimizer docs and
the profile-only code around:

- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/training/compact_policy_row_bridge.py`

I did not edit source, touch trainer defaults, or touch live training runs.

## Plain Current Amdahl Read

There are now two different bottleneck maps, depending on the denominator.

In the stock `train_muzero` denominator, the wall is still the LightZero
collect/search/object contract:

```text
host observations and timestep objects
-> GPU model calls
-> CPU root prep and CTree list APIs
-> Python simulation loop
-> recurrent outputs copied/listified back to CPU every simulation
-> per-env action dicts
-> stock replay, learner, and RND object lanes
```

That is why `direct_ctree_gpu_latent + output-fast` lands around `1.28x-1.31x`
in matched full-loop profile rows. It is a real improvement to the search
boundary, but it preserves the stock topology.

In the newer compact MCTX closed-loop denominator, search is no longer the
first wall. Search-boundary rows can hit tens or hundreds of thousands of
roots/sec, but the repeated loop falls back to much lower active roots/sec once
it has to step CurvyTron, update observations, rebuild root inputs, read actions
back, and write replay indices. The late H100 split says `env_step_sec` is
roughly `60-70%+` of wall in the strongest closed-loop rows, while actual game
mechanics are only about `8-11%` of that bucket. Most of that bucket is
state/observation handoff: actor render-state write, production-to-compact
conversion, renderer H2D/update, stack ownership, and synchronization.

Plain version:

```text
The current Amdahl wall is not "draw faster" and not "make MCTX faster."
It is that no single owner keeps CurvyTron state, renderer state, observation
stack, root batch, search result, and replay/RND sidecars in the layout the
next stage consumes.
```

## Why The Small Patches Got About 1.3x

The small patches were good patches, but they were local boundary cleanups.

- GPU-latent CTree kept hidden states on CUDA, but CTree still owned CPU tree
  state and consumed CPU/list reward, value, and policy payloads each
  simulation.
- The output-fast path mostly removed per-env action-output assembly. That
  bucket is now around noise in the matched no-RND direct row, so further
  output polish has little Amdahl room.
- Flat-A3 proved the Python/list ABI has cost in no-model microbenches, but the
  matched full-loop row did not improve because recurrent inference, root prep,
  stock collector shell, replay, learner, and RND remain.
- Dense Torch and compile spikes proved device tree arrays are directionally
  right, but eager/compiled Torch did not scale cleanly at sim16/sim32.
- Native actor buffer and compact sidecars removed a payload/merge layer, but
  closed-loop rows still spend most wall time feeding the next search input.
- CPU64 made the current CTree boundary slower, so the issue is not simply
  "more CPU cores."

So `~1.3x` is not mysterious. It is what happens when a patch accelerates a
large slice while keeping the old object ownership around it.

## Code-Level Ownership Read

`mctx_synthetic_benchmark.py` now has the useful closed-loop shape:

```text
HybridCompactBatch
-> build_compact_root_batch_v1(...)
-> JAX/MCTX search
-> validate_compact_search_result_v1(...)
-> manager.step(selected joint action)
-> build_compact_replay_index_rows_v1_from_search_result(...)
```

That is the right denominator, but it still pays root validation/copy work,
invalid-mask device transfer, action/readback synchronization, parent-manager
stepping, optional resident-stack FIFO update, and replay-index validation on
every loop.

`source_state_hybrid_observation_profile.py` has the right contract object,
`HybridCompactBatch`, and the profile-only native actor buffer already writes
reward, done, masks, joint actions, and render state into parent-owned arrays.
It also already has a `borrow_single_actor_render_state` canary. That canary is
useful, but it is only a probe. The bigger issue is that the renderer still
does not own the compact state it consumes across steps.

`compact_policy_row_bridge.py` is the strongest correctness scaffold in this
lane. `CompactRootBatchV1`, `CompactSearchResultV1`, and
`CompactReplayIndexRowsV1` keep row/player identity, legal masks, active roots,
`to_play=-1`, reward/done, terminal/final-observation facts, and search outputs
checked. The critique is not that the contract is too weak. The critique is
that the hot loop still materializes and validates too much of it at the wrong
owner boundary.

## Five Ambitious Architecture Changes

### 1. Persistent Compact Render-State Owner

Move from:

```text
actor production state -> parent render buffers -> renderer converts to compact state
```

to:

```text
actor/runtime updates renderer-native compact visual state in place
renderer consumes that compact state directly
resident stack/root batch borrows the result
```

This is not a small renderer tweak. It changes who owns visual state. The
persistent renderer should stop rebuilding compact policy-framebuffer input from
production state every step. Actor update should maintain the renderer's trail,
cursor, player, bonus, and live-prefix buffers directly, with stock production
state retained as validation/debug output.

Why it could matter: the current closed-loop wall is dominated by observation
handoff, not physics. Removing production-to-compact conversion, actor
render-state row copies, and some stack synchronization attacks the largest
measured bucket. Plausible gain is `1.5x-3x` on the current compact denominator,
and it is prerequisite plumbing for larger wins.

Small falsifier:

```text
Profile-only H100 B1024/P2/sim16/loop16:
  copied parent render-state baseline
  vs already-compact render-state input to persistent renderer

Pass: production_to_compact_sec and actor_render_state_write_sec mostly vanish,
total active roots/sec improves by at least 25-35%, and sampled host/resident
stack parity stays exact.

Kill: bucket movement without total wall improvement, or any mismatch in trail
cursor wrap, break markers, bonus state, avatar color, alive/dead rows, or
terminal final observation.
```

The existing `borrow_single_actor_render_state` mode is a useful warm-up
falsifier, but the real target should be renderer-native compact state, not just
borrowing the actor's production mapping.

### 2. Puffer-Style Native/Vector State Slab

Move CurvyTron physical rows into preallocated slabs:

```text
action[B,P]
state buffers
reward[B,P]
done[B]
legal_mask[B,P,3]
obs_stack[B,P,4,64,64]
sidecars for episode, alive, round, terminal, autoreset
```

Actors or worker chunks write directly into assigned slices. The parent does
not receive per-actor payload dataclasses and then merge/copy them back into a
new compact batch. This follows the useful PufferLib pattern: static memory,
contiguous buffers, worker-owned slices, no redundant observation copies, and
scalar compatibility output only at debug/validation edges.

Why it could matter: once scalar materialization is off, actor/env scheduling
and payload ownership are visible. A native actor buffer helped locally, but it
still sits above `VectorMultiplayerEnv` and parent-managed renderer state. A
slab owner can remove the next object layer and make B1024/B2048 the natural
shape instead of a copied aggregate.

Small falsifier:

```text
Local then H100 profile-only env driver:
  B512/B1024/B2048, P2, no scalar timestep, renderer off and persistent renderer on
  current HybridBatchedObservationProfileManager native_actor_buffer
  vs one slab owner that steps rows and writes masks/rewards/compact visual state in place

Pass: at least 2x improvement in env/update rows or at least 3x over the current
direct train-profile proxy when paired with mock/recurrent search; seeded scalar
parity for positions, rewards, legal masks, done, terminal/final rows.

Kill: performance within 20% of the current manager, or parity requires
reintroducing per-row Python objects in the measured loop.
```

### 3. Device-Resident Observation And Root-Batch Owner

Stop treating `[B,P,4,64,64]` as a host object that is rebuilt, copied, and
flattened for every consumer. The resident stack should be the hot source of
truth for search and RND latest-frame input, with host root batches used for
sampled validation.

Target shape:

```text
renderer writes latest frame on device
device FIFO updates [B,P,4,64,64]
root batch points at resident [B*P,4,64,64]
invalid masks and active-root masks are already device-ready
RND latest frame slices channel -1 on device
```

Why it could matter: resident mode already improves the repeated compact rows,
but the code still materializes host observations for validation and performs
explicit blocks that may serialize work unnecessarily. This change would remove
root-copy, stack-copy, and H2D pressure from the hot path instead of only
renaming them.

Small falsifier:

```text
Profile-only closed loop:
  resident stack, compact_root_copy_observation=false
  no hot-loop root observation copy
  invalid/action masks kept in a persistent device buffer
  sampled every-N validation copy only
  deferred resident-stack block, judged by total wall only

Pass: root_build_sec + h2d_sec + resident_stack_update_sec fall enough for
closed-loop active roots/sec to improve at least 30%, with sampled row-major
host/resident parity over FIFO shifts and reset rows.

Kill: synchronization simply moves into search_sec, or sampled validation finds
row/player or final-observation drift.
```

### 4. Real Device-Resident Search/Model Service

MCTX/JAX with a toy model proved search-boundary headroom. The next version has
to own the real model/search shape instead of bouncing through PyTorch host
callbacks or LightZero CTree lists.

Target shape:

```text
CompactRootBatchV1 device views
-> representation/prediction/recurrent in the same compiled/device runtime
-> MCTX or fixed-shape service-owned tree arrays
-> CompactSearchResultV1 arrays
```

This could be JAX/MCTX with translated/current weights, a fixed-shape
CUDA/Triton service, or a service-owned PyTorch/CUDA graph path if it actually
keeps the tree and recurrent loop device-resident. It cannot be a thin wrapper
that calls PyTorch recurrent inference and returns reward/value/policy to CPU
each simulation.

Why it could matter: direct CTree is capped by CPU/list search ownership.
MCTX showed `10x`-class search-boundary headroom, but search-only wins are
currently swallowed by env/observation. Once state ownership improves, search
will become hot again at sim32+ and for real model pressure.

Small falsifier:

```text
Closed compact loop, same B512/B1024 rows:
  toy MCTX
  vs real-shape/current-model device service
  vs direct_ctree_gpu_latent baseline

Pass: search+model bucket remains at least 3x faster than direct CTree at sim16
and sim32, no host callback per simulation, legal masks/visit policies/root
values pass CompactSearchResultV1, and total closed-loop speed improves after
state-owner costs are included.

Kill: real model bridge collapses to host callbacks or sim32 throughput falls
near current direct CTree.
```

### 5. Compact Replay/RND/Learner Owner

`CompactReplayIndexRowsV1` is the right collection-edge shape because it avoids
copying observations and next observations during collection. The bigger move
is to make replay, RND, and target construction consume compact records as
their native format, not as an adapter that eventually reconstructs thousands
of stock LightZero objects in the hot cadence.

Target shape:

```text
compact record ring
compact replay index rows
array-native target builder
RND latest-frame tensor ring
learner batch assembly from compact arrays
stock GameSegment/target rows only for sampled validation
```

Why it could matter: if actor/search gets faster, replay/RND/learner
materialization becomes the next Amdahl wall. The RND hash fix was a warning:
diagnostic/object lanes can erase search wins unless ownership changes too.

Small falsifier:

```text
Profile-only two-record then K-record compact replay/RND dry loop:
  selected actions from compact search drive next env step
  CompactReplayIndexRowsV1 writes into a ring
  target builder and RND latest-frame input read compact arrays
  scalar target rows materialized only for sampled parity

Pass: replay/RND/target edge stays under 10-15% of closed-loop wall at B1024,
and sampled target rows match the existing source-state target builder for
legal masks, final observations, rewards, done/terminated/truncated, to_play,
policy targets, and action history.

Kill: target/RND correctness forces per-root Python rows back into the measured
loop, or replay/RND becomes the largest bucket once env/search improve.
```

## What I Would Do First

Do the persistent compact render-state owner first.

Reason: the latest closed-loop Amdahl map says the wall is not game mechanics
and not MCTX search. It is the handoff that creates the next search input.
That makes state ownership the next honest target. A full native/vector state
slab is probably the larger destination, but it has more semantic surface area.
The renderer-native compact-state owner is the smallest experiment that attacks
the measured dominant bucket while keeping the current `HybridCompactBatch`,
`CompactRootBatchV1`, `CompactSearchResultV1`, and `CompactReplayIndexRowsV1`
contracts intact.

If it works, it gives the native/vector slab a clearer target: maintain exactly
the compact state the renderer/search/replay path already proved useful. If it
does not move total closed-loop wall, stop treating observation ownership as
the next multiplier and move straight to the native/vector slab or real
device-resident search/model service based on the measured bucket that remains.

## Guardrails For Any Big Move

- Keep all work profile-only until matched stock-loop A/B rows pass.
- Report physical rows, player roots, active roots, inactive roots, terminal
  roots, and padded roots in every roots/sec claim.
- Keep scalar LightZero objects as validation adapters, not hidden hot-path
  fallbacks.
- Validate terminal/final-observation and autoreset semantics before using
  normal-death rows as speed evidence.
- Judge async/JAX changes by total closed-loop wall, not by a bucket that may
  only have moved its synchronization wait elsewhere.
- Do not use search-boundary roots/sec as training-speed advice unless the
  repeated env/observation/replay/RND loop is in the denominator.
