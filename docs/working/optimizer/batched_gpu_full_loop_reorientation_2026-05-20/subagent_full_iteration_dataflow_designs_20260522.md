# Full Iteration Dataflow And Architecture Designs - 2026-05-22

Scope: read-only optimizer exploration over the current compact/profile
training-like loop. I did not edit source, start training, attach to live runs,
or change trainer defaults.

Guardrails:

- Trusted policy observation surface is `browser_lines + simple_symbols`,
  `[4,64,64]`, player perspective. The CPU oracle is the current trusted
  production/training surface.
- The fast optimizer renderer is
  `jax_gpu_persistent_policy_framebuffer_profile + direct_gray64` with
  browser-line and simple-symbol semantics, but it remains profile-only.
- The current MCTX path is profile-only. It does not call stock
  `train_muzero`, does not prove Coach training speed, and must not be promoted
  without separate semantic gates.

Latest profile anchors used here:

- Current persistent GPU renderer draw is already tiny in the loop24 rows:
  about `5-7ms` over the measured loop.
- In refresh-on loop24 rows, observation/search-input handoff is about
  `76-80%` of `env_step_sec`.
- Actual game mechanics are about `8-11%` of `env_step_sec`.
- Resident stack is now a real profile win after root-copy removal, about
  `1.3-1.4x` over the matching host-stack rows.
- Matched stock-loop direct hooks are still only about `1.28-1.31x` and remain
  profile evidence, not Coach defaults.

## Code Anchors

- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
  - `curvytron_hybrid_compact_visual_sample` setup and resident stack path.
  - `run_search(...)` JAX/MCTX compiled search.
  - closed compact loop buckets: `root_build_sec`, `h2d_sec`, `search_sec`,
    `d2h_sec`, `env_step_sec`, `replay_index_sec`.
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
  - `HybridBatchedObservationProfileManager`.
  - `InProcessHybridCurvyTronActor.step_into`.
  - `HybridCompactBatch`.
  - compact service replay proof helpers.
- `src/curvyzero/training/compact_policy_row_bridge.py`
  - `CompactRootBatchV1`.
  - `CompactSearchResultV1`.
  - `CompactReplayIndexRowsV1`.
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  - `_PersistentJaxPolicyFramebufferRenderer`.
  - direct CTree / array-boundary probes.
- `src/curvyzero/training/exploration_bonus.py`
  - current RND latest-frame extraction and CPU/device boundaries.

## One Compact Closed-Loop Iteration

This describes one measured loop iteration in the current compact MCTX profile
shape, using the common loop24 anchor as the mental model:

```text
B = physical rows, usually 1024 in latest loop24 rows
P = players per row = 2
R = roots = B * P
A = actions = 3
S = stack = [4,64,64]
```

1. The previous search result is converted to a host `joint_action[B,P]`.
   Active roots provide selected actions; inactive/terminal roots receive a
   default legal fill.

2. `HybridBatchedObservationProfileManager.step(joint_action)` starts the
   current `env_step_sec` bucket.

3. The manager validates `[B,P]` actions and partitions rows across in-process
   actors. With `native_actor_buffer=True`, actors are still stepped
   sequentially, but each actor writes into parent arrays instead of returning a
   payload object.

4. Each actor calls `VectorMultiplayerEnv.step(action_shard)`.
   This is CPU/NumPy game/env work. The strict mechanics leaves are runtime,
   reward, and post-runtime bookkeeping. Public action prep, public info, batch
   packing, final observations, and autoreset are adjacent but not pure physics.

5. In native mode, each actor writes compact sidecars directly:

```text
reward[B,P] float32
done[B] bool
episode_step[B] int32
elapsed_ms[B] float64
round_id[B] int32
alive[B,P] bool
action_mask[B,P,3] bool
joint_action[B,P] int16
terminal/autoreset row ids
```

6. If renderer-backed refresh is enabled, actors also write render-state rows
   into parent buffers. The persistent renderer keys include
   `visual_trail_pos`, `visual_trail_radius`, `visual_trail_owner`,
   `visual_trail_active`, `visual_trail_break_before`, `pos`, `radius`,
   `alive`, `present`, `avatar_color`, and bonus fields.

7. The manager updates the observation stack.

   Host-stack mode:

```text
shift host stack [B,P,4,64,64]
renderer.render(...)
GPU output -> host frames [B,P,1,64,64]
write latest host frame into stack
```

   Resident-stack mode:

```text
renderer.render(..., device_only=True)
renderer.last_output_device = [B,P,1,64,64] uint8 on JAX device
benchmark FIFO-updates resident stack on device:
  [B,P,4,64,64] <- concat(old[:, :, 1:], latest)
```

8. `_PersistentJaxPolicyFramebufferRenderer.render(...)` itself does:

```text
production render state -> compact render state
compact state -> delta state
delta/compose state host-to-device
persistent layer update on device
compose player-view gray64 frames on device
optional device-to-host readback
```

   The GPU draw/compose is small now. The larger work is state packing,
   render-state row writes, host-to-device setup, stack ownership, and sync.

9. The manager builds `HybridCompactBatch` with root identity sidecars:

```text
observation[B,P,4,64,64]
action_mask[B,P,3]
policy_env_id[R]
policy_env_row[R]
policy_player[R]
target_reward[R,1]
done_root[R]
to_play[R] = -1
active_root_mask[R]
final/terminal/autoreset masks
episode/action/reward sidecars
```

   In the MCTX profile, scalar LightZero timestep materialization is off. The
   compact capture probe receives the batch before the stock per-env object
   boundary.

10. The loop builds `CompactRootBatchV1`.

```text
observation[R,4,64,64]
legal_mask[R,3]
active_root_mask[R]
env_row[R]
player[R]
policy_env_id[R]
target_reward[R,1]
done_root[R]
```

   `copy_observation=False` in the latest no-copy rows avoids copying the host
   root observation. In resident mode, the host observation in this contract is
   primarily a metadata/validation edge; MCTX consumes the resident device stack.

11. The search input is readied on JAX device.

   Host-stack mode:

```text
jax.device_put(observation[R,4,64,64])
jax.device_put(invalid_actions[R,3])
block_until_ready() on both
```

   Resident-stack mode:

```text
observation_device = resident_stack.reshape(R,4,64,64)
jax.device_put(invalid_actions[R,3])
block_until_ready() on observation/mask
```

12. `run_search(...)` executes as a JIT-compiled JAX/MCTX call:

```text
uint8 visual stack -> float normalize
2-layer visual encoder -> hidden[R,H]
prediction -> prior_logits[R,3], value[R]
mctx.gumbel_muzero_policy(...)
recurrent_fn over fixed A=3 actions
invalid-action masking inside MCTX
block on action_weights
```

13. Search outputs are copied to host:

```text
action[R]
action_weights[R,3]
root_values[R]
```

   `CompactSearchResultV1` then validates only active roots: selected action is
   legal, visit policy is finite/non-negative/sums to 1, and root values are
   finite.

14. A new `joint_action[B,P]` is filled from the active-root search result.
   That action drives the next `manager.step(...)`.

15. `CompactReplayIndexRowsV1` is built from the previous compact batch,
   previous root batch, search result, and the next-step reward/done/action
   sidecars.

   It deliberately does not copy observations or next observations. It records
   indices and compact search/target arrays for a future sampler edge.

16. The loop records bucket times:

```text
root_build_sec
h2d_sec
search_sec
d2h_sec
env_step_sec
replay_index_sec
residual_sec
next_step_timings_sec
```

## Size And Movement Map

Concrete sizes below assume `B=1024`, `P=2`, `R=2048`, `A=3`.

| Object | Shape and dtype | Approx size | Move size | Read |
| --- | --- | ---: | ---: | --- |
| `joint_action` | `[1024,2] int16` | `4 KiB` | small host | Not important. |
| reward | `[1024,2] float32` | `8 KiB` | small host | Not important. |
| done | `[1024] bool` | `1 KiB` | small host | Not important. |
| action mask | `[1024,2,3] bool` | `6 KiB` | small host/H2D | Tiny but synchronizes search input. |
| compact identity arrays | `R int32/int64` | `8-16 KiB` each | small host | Validation/replay metadata. |
| latest frame | `[1024,2,1,64,64] uint8` | `8 MiB` | large if D2H | Avoided for MCTX input in resident mode. |
| full stack uint8 | `[1024,2,4,64,64] uint8` | `32 MiB` | large H2D if host stack | The main observation payload. |
| full stack float32 | same | `128 MiB` | very large | Older/stock-like materialization is much worse. |
| host stack shift | write `3/4` of stack | `24 MiB` uint8 | large host copy | Avoid with resident/device-owned stack. |
| root observation copy | `[R,4,64,64] uint8` | `32 MiB` | large host copy | Latest no-copy removes this local copy. |
| visual trail pos | `[1024,4096,2] float32` | `32 MiB` | large host row write | Doubles if float64. |
| visual trail radius | `[1024,4096] float32` | `16 MiB` | large host row write | Doubles if float64. |
| visual trail owner | `[1024,4096] int32` | `16 MiB` | large host row write | Full-row render-state writes are costly. |
| visual active/break | two `[1024,4096] uint8` | `8 MiB` | large host row write | Also scanned for deltas. |
| renderer output device | `[1024,2,1,64,64] uint8` | `8 MiB` | device resident | GPU draw is small; ownership around it is not. |
| MCTX hidden | `[2048,64] float32` | `512 KiB` | device only | Small compared with image stack. |
| MCTX latent/node pool | roughly `[R,sim+1,H]` | `~8.5 MiB` at sim16/H64 | device only | Search-internal, not the current wall. |
| action output | `[2048] int32` | `8 KiB` | small D2H | Needed before next CPU env step. |
| action weights | `[2048,3] float32` | `24 KiB` | small D2H | Needed for replay/validation, not for env step. |
| root values | `[2048] float32` | `8 KiB` | small D2H | Needed for replay/validation. |
| compact replay index rows | active roots, mostly scalar arrays | `<128 KiB` | small host | Current proof cost is not the wall. |
| RND latest input | `[R,1,64,64]` uint8/float32 | `8/32 MiB` | potentially large | Current RND path CPU-normalizes and copies. |

Large moves today:

- Actor render-state row writes and full visual-trail arrays.
- Production-to-compact and delta construction around the persistent renderer.
- Host stack shift/latest update when host stack is active.
- GPU renderer output readback in host-stack mode.
- Host stack H2D into MCTX in host-stack mode.

Small moves today:

- Action masks, selected actions, action weights, root values.
- Compact replay index rows.
- Reward/done/action sidecars.

Important nuance: a small array can still create a large wall if it forces a
device synchronization at a bad point.

## Current Sync Points

Hard or effectively hard syncs in the compact MCTX loop:

- Persistent renderer update: `_layer_device.block_until_ready()` after the
  layer update.
- Persistent renderer compose: `output_device.block_until_ready()` after
  composing `[B,P,1,64,64]`.
- Host-stack readback: `np.asarray(output_device)` when `device_only=False`.
- Resident stack update: `compact_visual_resident_device_stack.block_until_ready()`.
- Search input readiness: `obs.block_until_ready()` and
  `invalid_actions.block_until_ready()` before search timing.
- MCTX search completion: `loop_output.action_weights.block_until_ready()`.
- Search output readback: `np.asarray(loop_output.action)`,
  `np.asarray(loop_output.action_weights)`, and root-value extraction.
- Profile timers deliberately force readiness so bucket attribution is honest.

Other syncs in neighboring train/profile paths:

- Torch direct CTree root prep copies `value`/`policy_logits` to CPU.
- Direct CTree GPU-latent search copies recurrent `reward/value/policy` outputs
  to CPU every simulation for CTree backprop.
- RND uses `detach().cpu().numpy()`, `.cpu().item()`, model-state hashes, and
  list-backed replay samples.

## Syncs That Could Be Delayed

- Do not block renderer compose immediately if the next consumer is another
  JAX op. Let resident stack update and search inherit the dependency.
- Do not block a resident stack reshape before `run_search`; the JIT call can
  consume the device array directly.
- Move invalid-action mask H2D earlier or leave it async until search consumes
  it. It is tiny but currently participates in the boundary timing.
- Read back only selected `action` before the next CPU env step. Delay
  `action_weights` and `root_values` until replay-index construction, or batch
  them with validation every N profile steps.
- Build replay rows after the next env step while the next search input is
  being prepared, if action dependencies are kept clear.
- Move RND hashes, metric scalar reads, and diagnostics off the per-collect hot
  cadence.

These are not guaranteed speedups. They are falsifiers for whether sync
placement, rather than raw computation, is inflating the handoff bucket.

## Architecture Options

### 1. Tighten The Current Compact Loop

Design:

- Keep the current manager, persistent renderer, MCTX search, and compact
  replay contracts.
- Keep `copy_observation=False`.
- Avoid building any host root observation fields that are not used for
  validation.
- Pack selected action, weights, and root values in one output readback.
- Reduce per-step validation in trusted profile rows while keeping sampled
  parity checks.

Expected speedup:

- `1.05-1.15x` over the latest no-copy resident row.
- Less if no-copy already removed the main local copy.

Critique and risks:

- This cannot attack the `76-80%` observation handoff wall deeply.
- Easy to get a prettier profile without changing the architecture.
- Must preserve legality, active-root masks, and root ordering.

When to stop:

- If loop24 roots/sec moves less than `5%`, stop polishing this layer.

### 2. Renderer-Owned Compact State / Delta Handoff

Design:

- Stop copying full actor render-state rows into parent render buffers every
  step.
- Give the persistent renderer the compact layout it already needs, or let
  actors write directly into a renderer-owned compact/delta buffer.
- Keep the same `browser_lines + simple_symbols` policy-space semantics.
- Keep host compact sidecars for reward/done/mask/replay validation.

Expected speedup:

- Plausible `1.2-1.6x` over current resident loop24 rows if render-state write
  and production-to-compact are the true wall.
- Could be higher in long-trail/no-death rows; lower in normal-death rows.

Critique and risks:

- This touches the most suspicious current bucket.
- Reset/autoreset ordering is subtle: final observation must be captured before
  reset rows are zeroed or re-rendered.
- A delta bug can create visually stale but legally valid search roots.
- The CPU oracle remains the trusted surface, so parity sampling is required.

### 3. Harden Resident Device Stack As The Search Input

Design:

- Treat `renderer.last_output_device -> resident FIFO stack -> MCTX` as the
  normal profile input path.
- Host `CompactRootBatchV1` supplies identity, masks, and replay metadata, not
  the hot observation tensor.
- Add sampled parity checks between resident stack rows and host/CPU oracle
  rows at warmup and reset boundaries.

Expected speedup:

- Already observed: about `1.31x` at sim16 and `1.38x` at sim32 in matching
  loop24 host-vs-resident rows.
- Incremental hardening may add little speed, but it converts a speed probe into
  a trustworthy baseline for deeper work.

Critique and risks:

- The current host root observation can be a zero/validation placeholder in
  resident mode, so correctness can look fine while search consumed stale
  pixels.
- Row-major `[env row, player]` order must stay pinned.
- Terminal/final-observation and autoreset parity are the danger zones.

### 4. Delayed-Readback And Double-Buffered Replay Edge

Design:

- Read back only `action[R]` immediately.
- Delay `action_weights[R,3]` and `root_values[R]` until after the next
  `manager.step(...)`, or accumulate several steps before validation/replay
  materialization.
- Double-buffer compact batches and search results so CPU env stepping can
  overlap with small D2H/replay work where dependencies allow.

Expected speedup:

- Likely `1.05-1.25x` in current loop24 rows.
- More useful after observation handoff shrinks and search/replay become larger
  fractions.

Critique and risks:

- Actions are a real dependency for the next CPU env step, so this cannot
  overlap the whole search.
- Delaying validation makes failures farther from their cause.
- Replay row ordering and record indices must remain deterministic.

### 5. Compact Replay And RND Resident Adapters

Design:

- Keep `CompactReplayIndexRowsV1` in the collect hot path.
- Add a compact sampler edge that materializes learner/RND tensors only on
  sample, not during collection.
- Feed RND latest-frame input from `[B,P,4,64,64]` compact/resident stacks,
  preferably `[R,1,64,64]` on device.
- Move hashes and scalar metrics off the hot cadence.

Expected speedup:

- Small in current no-RND rows: `1.0-1.1x`.
- In RND rows, likely `1.05-1.25x` if CPU latest-frame extraction and metrics
  reappear as a wall.
- Strategically required before any compact path is Coach-facing.

Critique and risks:

- RND is not why current direct search is only `1.3x`.
- Reward-shaping semantics are high risk: meter mode, target-reward mode,
  normalization, hashes, and metrics must remain explainable.
- If implemented early, it may distract from the larger observation handoff.

### 6. Array-Native Fixed-A3 CTree / Search Service Boundary

Design:

- Keep the current PyTorch model initially.
- Replace Python/list CTree edges with fixed `A=3` arrays:

```text
prepare(value[R], policy[R,3], legal[R,3])
traverse() -> path_index[R], action[R]
backprop(reward[R], value[R], policy[R,3])
output() -> action[R], visit_policy[R,3], root_value[R]
```

- Use compact roots and compact replay rows as the ABI.

Expected speedup:

- Plausible `1.3-2.2x` over current direct CTree profile rows.
- Not a standalone `5-10x` full-loop plan unless the env/observation wall is
  also cut.

Critique and risks:

- Flat-A3 alone already failed to move matched full-loop speed enough.
- The useful version must remove ownership boundaries, not only listify cost.
- Dirichlet noise, min/max stats, root values, legal masks, and visit
  distributions need fixed-seed parity gates.

### 7. JAX/MCTX Compact Search Owner With Current-Model Realism

Design:

- Keep compact env/observation/replay contracts.
- Let JAX/MCTX own tree/search arrays on device.
- Replace the toy JAX model with either a validated JAX port/export of the
  current model or a documented replacement model lane.
- Keep search outputs in compact arrays until replay/learner edges.

Expected speedup:

- Search-only headroom is already 10x-class.
- Repeated compact loop speed depends on the handoff. If options 2 and 3 land,
  a `2-4x` profile loop improvement is plausible.
- A `5-10x` training claim remains unproven until replay/RND/learner are in the
  denominator.

Critique and risks:

- Toy MCTX does not prove current MuZero training.
- Model parity and training-target semantics are major work.
- JAX/Torch interop can create new syncs unless one side owns the loop.

### 8. Native Compact Env State Owner

Design:

- Move below `VectorMultiplayerEnv.step(...)` public packaging.
- Own preallocated compact row/player buffers for runtime state, reward, done,
  masks, render deltas, and replay sidecars.
- Public `BaseEnvTimestep` and `MultiplayerTrainerStepV0` become validation or
  compatibility adapters, not the hot path.
- Implementation could start as NumPy/C++/Numba before any GPU env rewrite.

Expected speedup:

- If it halves the current `env_step_sec`, total loop speed improves roughly
  `1.5-1.7x` in rows where env is `60-70%` of wall.
- If it cuts env/observation handoff by `4x`, total loop speed can reach
  `2-3x` before search/learner become the next wall.

Critique and risks:

- This is the first option that directly addresses the dominant bucket.
- It is also where game semantics can drift: death, autoreset, trail cursors,
  masks, reward timing, and final observations.
- The current profile uses no-death shapes heavily; normal-death validation is
  mandatory before broad claims.

### 9. Replacement Compact Trainer Topology

Design:

```text
compact actors/state owner
-> resident policy observation / root buffer
-> batched search service
-> compact replay writer and sampler
-> learner/RND device or array adapters
```

- Stock LightZero remains a semantic oracle and comparison baseline.
- Coach-facing use requires a documented replacement for stock `train_muzero`,
  not a hidden config tweak.

Expected speedup:

- This is the only credible `5-10x` class design.
- It needs multiple wins to compose: observation/state ownership, search owner,
  replay/RND arrays, and learner input ownership.

Critique and risks:

- Highest implementation cost.
- Highest semantic risk.
- The payoff is large only if the measured compact profile survives normal
  death/autoreset, RND, learner sampling, checkpoint/resume, and tournament
  surface gates.

## Recommendation

Do not spend the next cycle on renderer draw kernels, flat-A3 polishing, or
CPU-count tuning. The current loop says the draw is already tiny, game
mechanics are not the wall, and compact replay-index rows are cheap.

The next architecture should attack ownership of the observation/search-input
handoff:

```text
actor/runtime state
-> compact render/delta owner
-> resident stack/root input
-> MCTX/search
```

Use current compact contracts as the validation spine:

- `HybridCompactBatch` for row/player truth.
- `CompactRootBatchV1` for root identity and masks.
- `CompactSearchResultV1` for legality and search outputs.
- `CompactReplayIndexRowsV1` for no-observation-copy replay writes.

## Recommended Next Two Falsifiers

### Falsifier 1: Renderer-State Ownership Kill Test

Question:

```text
If actor render-state row copies and production-to-compact ownership are
removed or sharply reduced, does loop24 resident throughput move?
```

Profile-only shape:

- H100.
- `B1024/P2/body4096/h64/depth16/loop24/native/no-copy/replay`.
- Run sim16 and sim32.
- Use resident stack input.
- Keep trusted policy semantics: browser-lines plus simple-symbols.
- Keep compact root/search/replay validation.

Candidate intervention:

- Renderer-owned compact/delta buffer, borrowed render state, or any canary
  that sharply reduces `actor_render_state_write_sec` and
  `renderer_production_to_compact_sec` without changing search.

Keep condition:

- At least `1.25x` over the current resident row, or
- observation handoff falls from about `76-80%` of `env_step_sec` to below
  `55%` while legality/replay checks stay green.

Kill condition:

- Less than `10%` throughput gain and the handoff fraction remains above
  `65%`.

Why this first:

- It directly tests the largest named wall after GPU draw became tiny.

### Falsifier 2: Training-Like Normal-Death/RND Compact Edge

Question:

```text
Does the resident compact loop remain dominated by observation/search-input
handoff when profile-no-death assumptions are relaxed and RND input is present?
```

Profile-only shape:

- Same resident/no-copy compact MCTX loop as the best current baseline.
- Add a normal-death/autoreset variant.
- Add an RND latest-frame input canary from compact `[B,P,4,64,64]` without
  scalar timestep materialization.
- Keep replay-index rows on.

Keep condition:

- Resident stack still beats host stack by at least `1.2x`.
- Observation/search-input handoff remains the largest `env_step_sec` leaf.
- RND input extraction adds less than `10-15%` wall.
- Terminal/autoreset/final-observation parity checks pass.

Kill condition:

- Normal death/reset makes trails short enough that render-state ownership is no
  longer material, or
- RND/latest-frame extraction becomes the new largest wall, or
- parity around terminal/final observations fails.

Why this second:

- It prevents overfitting the architecture to no-death loop24 profiles before
  spending a large implementation budget.

