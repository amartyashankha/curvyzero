# GPU/CPU Sync And Residency Strategy, 2026-05-22

Scope: CurvyTron compact closed-loop optimizer path, especially the H100
renderer-backed resident/host stack rows. This is a design note only. I read the
working docs plus `source_state_hybrid_observation_profile.py` and
`source_state_batched_observation_boundary_profile.py`; I did not edit source,
launch jobs, or touch live runs.

## Plain Read

The fresh H100 rows changed the question from "is search fast enough?" to "why
does the next search input require so much synchronized host work?"

The important current read is:

```text
B1024/P2/body4096/h64/depth16/loop24/native/no-copy/replay:
  host sim16:     23.1k roots/sec, env 68.1%, search 7.5%
  resident sim16: 30.3k roots/sec, env 68.3%, search 9.6%
  host sim32:     19.5k roots/sec, env 63.0%, search 15.2%
  resident sim32: 26.8k roots/sec, env 59.8%, search 19.7%
  refresh-off ceiling sim16: 57.9k roots/sec

Inside env_step_sec on refresh-on rows:
  actual game mechanics: about 8-11% of env_step_sec
  observation/search-input handoff: about 76-80% of env_step_sec
  GPU draw: about 5-7ms over the measured loop
```

Resident stack is a real direction, roughly `1.3x-1.4x` in the matched rows, but
it only removes part of the image bounce. The remaining wall is state ownership
and synchronization around the renderer:

```text
CPU actor state
-> parent render-state copy/write
-> compact/delta pack
-> renderer H2D
-> persistent update/render synchronization
-> host stack or resident stack update
-> root/search input handoff
```

So the next strategy should not be another search-kernel polish. It should make
the observation/search-input handoff a resident dependency chain, with host
materialization only at semantic commit points.

## Syncs That Happen Now

Current profile loop, simplified:

```text
CPU action from previous search
-> VectorMultiplayerEnv.step(...)
-> write compact scalar sidecars
-> write/copy render state into parent buffers
-> persistent JAX renderer
-> host or resident stack update
-> compact root/search input
-> MCTX/JAX or LightZero search
-> action/readback for next CPU step
-> replay-index/target sidecars
```

Concrete sync and copy sites visible in the inspected source:

| Site | Current behavior | Why it syncs or copies |
| --- | --- | --- |
| Actor render-state write | `InProcessHybridCurvyTronActor.step_into(...)` calls `_write_native_render_state_rows(...)` when renderer-backed native buffers are enabled. | CPU-to-CPU copy of actor `env.state` fields into parent-sized render buffers. Hot because it walks visual trail/player/bonus arrays before every render. |
| Host stack update | `_update_observation(...)` shifts `self._zero_stack`, calls `observation_renderer.render(...)`, validates `np.asarray(result.frames)`, and writes the latest frame into the host FIFO stack. | In host mode this forces frames to be host-visible and then rebuilds `[B,P,4,64,64]` for the next consumer. |
| Persistent renderer input | `_PersistentJaxPolicyFramebufferRenderer.render(...)` builds compact state, packs delta state, copies delta/compose state to JAX device, blocks after persistent update, blocks after compose, and optionally reads `np.asarray(output_device)`. | The GPU draw itself is small now; the synchronized CPU pack/H2D/update/compose/readback envelope is hot. |
| Resident stack update | In resident compact visual mode, docs show the renderer runs `device_only=True`, then the benchmark maintains a separate JAX stack from `last_output_device` and explicitly blocks. | Avoids renderer D2H and search obs H2D, but adds a second stack owner and a forced timing barrier. |
| Batched stack probe / LightZero input | `_prepare_observation_tensor(...)` turns host `flat_stack` into Torch CUDA tensors, optionally pins, normalizes, and synchronizes after transfer/normalization. The `resident_torch_reuse` mode is a stale-input ceiling. | Necessary for the current LightZero host-stack input path; not a valid fresh training path when reuse is stale. |
| Direct CTree root prep | `_run_direct_mcts_arrays(...)` syncs after initial inference, then copies root value/policy logits to CPU NumPy and listifies logits/legal actions/noise for CTree roots. | Required by the current CPU/list LightZero CTree API. Not the main compact MCTX wall, but important if the direct CTree lane remains. |
| GPU-latent CTree recurrent loop | `_run_direct_ctree_gpu_latent_search(...)` keeps latents on device, but each simulation still gets Python traverse output, copies action indices H2D, runs recurrent inference, then copies reward/value/policy logits back to CPU lists for backprop. | Required by current CTree semantics. Removing it requires an array-native or GPU-resident tree contract. |
| Search output readback | Compact closed loop must read selected actions before the next CPU env step. Replay sidecars also need policy/value/visit outputs before the current CPU replay writer consumes them. | Action readback is semantically necessary while env mechanics are CPU. Replay output readback can often be chunked. |
| Profiling syncs | Torch `_sync_torch_device_if_cuda(...)`, JAX `block_until_ready()`, and timing readbacks are used to make bucket attribution honest. | Useful measurement, but not always an algorithmic barrier if total wall is measured elsewhere. |

## Necessary Syncs For Semantics

These barriers should be treated as real until the architecture changes:

1. **Action barrier before CPU env step.** The CPU env cannot step row `t+1`
   until selected actions from search over observation `t` are known.
2. **CPU sidecar barrier before search.** Legal masks, done flags, rewards,
   active-root masks, row/player ids, and `to_play` must correspond to the same
   post-step state as the observation searched.
3. **Terminal/final-observation barrier.** If a row terminates, the final
   observation must be captured before autoreset mutates the row. A resident
   path needs an affected-row device or host snapshot at exactly that point.
4. **Renderer-to-search dependency.** Search must consume the latest frame after
   the renderer update for the same step. This requires a device dependency, but
   not necessarily a host `block_until_ready()` immediately after render.
5. **Replay/target commit barrier.** Search policy, root value, action, reward,
   done, action mask, env row, player, and final-observation sidecars must be
   available before the learner-facing replay/target writer commits the sample.
   This can be a chunk boundary, not necessarily every env step.
6. **Validation/checkpoint/summary barriers.** Exact parity checks, sampled host
   mirrors, debug summaries, and checksums may synchronize, but they should be
   explicitly sampled or marked as profile/debug overhead.

The key distinction: action and terminal ordering are semantic barriers.
Per-step full-stack host visibility is not.

## Syncs To Delay, Amortize, Or Delete

Highest-value candidates:

1. **Per-step observation D2H.** In resident mode, do not read the full rendered
   frame stack back just to feed search. Keep `last_output_device` and the
   resident FIFO stack as the hot search input. Use host readback only for
   sampled parity, terminal rows, or debug.
2. **Per-step host stack FIFO.** When scalar materialization is disabled and the
   consumer is device-native, the host `[B,P,4,64,64]` stack should become a
   sampled mirror, not the source of truth.
3. **Immediate resident-stack `block_until_ready()`.** Let the next search
   consume the resident stack and provide the dependency. Measure total loop
   wall, because removing this block may simply move wait time into search.
4. **Root observation copy in compact root build.** If search consumes a device
   stack handle, do not copy the full host observation into `CompactRootBatchV1`
   on the hot path. Keep metadata and sampled observation parity.
5. **Actor render-state parent copy.** The native actor buffer removed payload
   objects, but `actor_render_state_write_sec` remains hot. For `actor_count=1`,
   borrow actor state directly. Longer term, actors should maintain the compact
   renderer layout, not rebuild it from production state every step.
6. **Renderer compact/delta pack H2D.** The persistent renderer still does CPU
   compact-state conversion and delta packing each render. Move ownership of the
   compact renderer state closer to the actor/env step so the renderer receives
   already-compact deltas.
7. **Replay/RND readbacks and metrics.** Visit policies/root values/RND latest
   frames can be staged in compact arrays and flushed in chunks. CPU hashes and
   summary metrics should not sit in the per-step action-critical path.

Lower priority in this compact MCTX denominator:

- Search-kernel speedups. Search is usually single-digit to about `20%` in the
  fresh rows, while env/observation handoff is the majority.
- Full GPU env rewrite. Actual game mechanics are only `8-11%` of the current
  `env_step_sec`; moving all physics first is too large a change for the next
  falsifier.

## Target Architecture

Use a two-owner compact loop:

```text
CPU owner:
  action application
  CurvyTron mechanics
  reward/done/legal-mask sidecars
  terminal/autoreset decisions
  small selected-action readback

GPU owner:
  persistent trail framebuffer
  latest rendered frame
  resident [B,P,4,64,64] uint8 FIFO
  search input observation
  optional staged search/replay tensors
```

The hot step should look like:

```text
1. CPU env applies selected actions and mutates compact scalar sidecars.
2. Actor/env emits compact renderer deltas or borrowed render state.
3. Persistent renderer updates device layer and produces latest device frame.
4. Resident device FIFO appends latest frame.
5. Root builder supplies only small host sidecars: mask, reward, done, ids.
6. Search consumes resident device observation plus sidecars.
7. Only selected actions read back before the next CPU env step.
8. Replay/target/RND outputs flush in compact chunks, with sampled validation.
```

Robustness rules:

- Keep CPU mechanics authoritative for now. The goal is not a GPU env rewrite.
- Keep row/player/root identity host-visible and cheap.
- Treat terminal rows as special. If exact final-observation ordering is not
  proven, fall back to the current copied path for those rows.
- Make the hot path static-shape: fixed `B`, `P=2`, stack depth, `64x64`,
  action count `3`.
- Preserve a debug host mirror, but sample it. For example: warmup, first N
  steps, every K steps, and all terminal/autoreset rows.
- Optimize only total closed-loop wall. Bucket movement caused by less
  synchronization is not a win unless roots/sec moves.

## Smallest Falsifier Experiments

### 1. Borrowed Or Already-Compact Render State

Question: is the next wall the actor render-state copy and production-to-compact
conversion?

Small slice:

```text
Profile-only, actor_count=1, no-terminal or terminal-fallback mode:
  copied parent render-state baseline
  vs borrowed actor env.state or already-compact renderer-state input
```

Keep the same resident/host stack mode, same actions, same root no-copy mode,
and same replay-index setting. Report `actor_render_state_write_sec`,
`renderer_production_to_compact_sec`, `renderer_persistent_delta_pack_sec`,
total closed-loop roots/sec, and sampled observation parity.

Falsifies the ownership thesis if `actor_render_state_write_sec` collapses but
total roots/sec improves by less than about `10%`, or if parity fails for
row/player, legal mask, reward/done, latest frame, or terminal fallback.

### 2. Deferred Resident Stack Synchronization

Question: is resident mode paying an avoidable explicit barrier, or is the wait
required by the next search anyway?

Small slice:

```text
Current resident stack:
  render device_only
  update resident FIFO
  block_until_ready
  search

Lazy resident stack:
  render device_only
  enqueue FIFO update
  no immediate block
  search consumes FIFO and becomes the dependency
```

Read total wall and roots/sec only. Also report where the wait moved:
`resident_stack_update_sec`, `h2d_sec`, `search_sec`, and residual.

Falsifies this as a win if total roots/sec does not improve, even if the
resident-stack bucket shrinks.

### 3. Hot Device Observation With Sampled Host Mirror

Question: can the host observation stack and root observation copy leave the hot
path without breaking semantics?

Small slice:

```text
Resident device stack is the search observation.
Host CompactRootBatch keeps sidecars only.
Host full observation is built on a sampled cadence and for terminal rows.
```

Compare against current resident no-copy root rows. Require sampled parity of
device stack versus host stack over FIFO shifts, row/player reshape, first-frame
warmup, and reset/final-observation cases. Report observation D2H bytes, host
stack bytes, root observation copy bytes, mask H2D bytes, and total roots/sec.

Falsifies the residency path if either parity is fragile or throughput does not
move at least `10-15%` after deleting the host stack/root-observation work from
the hot cadence.

## Recommendation

Do not broaden the system yet. The compact loop already proves search headroom,
and the H100 rows say the current bottleneck is the observation/search-input
handoff. The next robust architecture is CPU mechanics plus GPU-resident visual
and search input, with synchronization only at action, terminal, and replay
commit points.

Start with borrowed/already-compact render state, then lazy resident-stack
sync, then sampled host-mirror removal. Those three falsifiers are small enough
to kill quickly, and together they test the real hypothesis: whether making the
renderer/search input resident can close the gap between the current `23k-30k`
refresh-on rows and the `57.9k` refresh-off ceiling.
