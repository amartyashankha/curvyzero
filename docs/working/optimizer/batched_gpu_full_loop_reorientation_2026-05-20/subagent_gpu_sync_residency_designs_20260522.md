# GPU Sync / Residency Design Memo, 2026-05-22

Scope: compact H100 visual-profile loop, especially
`curvytron_hybrid_compact_visual_sample` with persistent JAX framebuffer,
native actor buffer, root no-copy, replay rows, and optional resident visual
stack. This is a design memo only. I read code/docs and did not launch or touch
live training runs.

## Short Read

The latest H100 rows say the current wall is not CurvyTron mechanics and not raw
GPU drawing. It is the observation/search-input handoff.

Measured anchor rows, all B1024/P2/body4096/h64/depth16/loop24/native/no-copy
with replay rows:

| row | run | roots/sec | env frac | search frac | mechanics/env | obs handoff/env | GPU draw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| host stack, sim16 | `ap-iHy06LxrTZNDIPX0soLPwJ` | `23,109` | `68.1%` | `7.5%` | `9.6%` | `80.0%` | `0.0066s` |
| resident stack, sim16 | `ap-HGSnFYPldyzmgSjQedA9CK` | `30,297` | `68.3%` | `9.6%` | `8.4%` | `79.2%` | `0.0054s` |
| host stack, sim32 | `ap-fD9oGLwDmrnakTogszpz9W` | `19,485` | `63.0%` | `15.2%` | `8.8%` | `79.8%` | `0.0071s` |
| resident stack, sim32 | `ap-CTD65oO51yoHPexUwvA5JQ` | `26,805` | `59.8%` | `19.7%` | `10.8%` | `75.8%` | `0.0057s` |
| refresh-off ceiling, sim16 | `ap-aBw0riUkgxyj97vyuhtUVA` | `57,895` | `26.1%` | `18.5%` | `43.1%` | `0.0%` | skipped |

The follow-up resident retest after root no-copy is also directionally clear:
resident sim16 `31,611` vs host `26,610` class rows, and resident sim32
`28,855` vs host sim32 `21,167`. Resident stack helps once the root observation
copy is removed, but env/observation handoff is still the Amdahl wall.

Additional follow-up after borrowed render-state and root no-copy:

| row | run | sims | resident sync | roots/sec | env frac | search frac | read |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| resident borrowed | `ap-Mgvb8AFe3q1HzbiMAepQjL` | 16 | on | `48,579` | `53.8%` | `15.2%` | sync baseline |
| resident borrowed | `ap-iA7EpkLNZLvCPjMDEWodxi` | 16 | off | `54,717` | `52.4%` | `16.7%` | small win |
| resident borrowed | `ap-IkSystwnJLV5kDHSFobmZe` | 32 | on | `36,041` | `43.3%` | `28.3%` | sync baseline |
| resident borrowed | `ap-QKJXlY0dZ0zIQpJU4g4oA6` | 32 | off | `27,944` | `44.4%` | `24.4%` | regression |

Plain read: lazy resident-stack sync is useful as an attribution switch, but it
is not a promotion by itself. At sim16 it buys a small same-denominator win; at
sim32 it loses badly enough that the wait probably moved, interacted with
asynchronous work, or exposed noise. The next prototype should attack the
larger remaining leaves directly: compact-state packing, H2D/update ownership,
public packaging, and the search boundary.

## Current Hot Loop Dataflow

### Host-owned large state

- CPU `VectorMultiplayerEnv.state` is still authoritative for mechanics:
  positions, alive/present, timers, bonuses, action masks, visual/body trail
  state, done/reward, autoreset state.
- `HybridBatchedObservationProfileManager` owns host compact sidecars:
  reward `[B,P]`, done `[B]`, alive/action_mask `[B,P,...]`, row/player ids,
  joint action, terminal/autoreset rows.
- In host-stack mode it also owns the live observation stack
  `[B,P,4,64,64]`, currently `uint8` or `float32`. At B1024/P2/uint8 this is
  about 32 MiB per stack; the latest frame is about 8 MiB.
- `CompactRootBatchV1` is host/NumPy. With `copy_observation=False`, root
  observation is a reshape/view of the stack; masks/ids/reward/done are still
  copied small arrays.

### GPU-owned large state

- `_PersistentJaxPolicyFramebufferRenderer` owns the persistent trail layer on
  JAX device and produces `last_output_device` shaped `[B,2,1,64,64]` `uint8`.
  This is the fresh frame the next search wants.
- In resident mode, `mctx_synthetic_benchmark.py` maintains a separate JAX FIFO
  stack `[B,2,4,64,64]` by concatenating `device_stack[:, :, 1:]` with
  `renderer.last_output_device`.
- MCTX search consumes JAX device observations `[B*2,4,64,64]` plus an invalid
  action mask. Search outputs selected action, visit policy/action weights, and
  root values on device until read back.

### Small host/device traffic

- Legal/action masks are `[B,2,3]` bool, only kilobytes at current B.
- Rewards/done/to_play/env_row/player/policy_env_id/root masks are small
  sidecars. They should remain host-visible while CPU mechanics are
  authoritative.
- Selected actions are tiny (`B*P` int-ish values). This readback is a real
  semantic barrier because the CPU env needs actions for the next step.
- Visit policies/root values are small relative to observations, but can still
  matter if flushed every step with synchronization.

### Step order today

```text
search output on GPU
-> D2H selected actions/action weights/root values
-> CPU joint action [B,2]
-> InProcessHybridCurvyTronActor.step_into(...)
   -> CPU VectorMultiplayerEnv.step(...)
   -> write compact sidecars into parent NumPy buffers
   -> write render-state rows into parent render buffers
-> manager._update_observation(...)
   -> optional host stack shift
   -> renderer.render(...)
      -> CPU production/native render state -> compact state
      -> CPU persistent delta pack
      -> H2D delta/compose state
      -> JAX persistent framebuffer update
      -> JAX compose latest frame
      -> optional D2H latest frames
   -> optional host latest-frame stack write
   -> terminal final_observation/autoreset reset-frame handling
-> HybridCompactBatch/CompactRootBatchV1 host sidecars
-> if host stack: H2D full root observation + mask
-> if resident stack: use resident JAX stack + H2D mask
-> MCTX search
-> compact replay-index rows
```

The code surfaces matching this are:
`InProcessHybridCurvyTronActor.step_into`, `_write_native_render_state_rows`,
`HybridBatchedObservationProfileManager._update_observation`,
`_PersistentJaxPolicyFramebufferRenderer.render`, and the closed-loop block in
`mctx_synthetic_benchmark.py`.

## Likely Implicit Sync Points

These are the places where the current profile either forces readiness or makes
host/device data semantically visible:

| Site | Data | Size | Why it syncs or copies |
| --- | --- | ---: | --- |
| Search output readback | actions, action weights, root values | small | CPU env cannot step until selected actions exist. Visit/root arrays are replay/validation outputs. |
| Actor compact sidecar write | reward/done/action mask/ids | small | CPU NumPy owner writes current truth after mechanics. |
| Actor render-state write | visual trail/player/bonus render buffers | large-ish CPU copy | Native path still copies actor-local env state into parent render buffers every step; recent rows show this is hot. |
| Renderer production-to-compact | render state -> compact fields | medium CPU work | Persistent renderer converts from production/native arrays before delta pack. |
| Renderer delta/compose H2D | trail deltas, reset masks, compose state | medium | Required for device framebuffer update while env state is CPU. |
| Renderer update/compose blocks | persistent layer, latest frame | large device state | `block_until_ready()` is used for attribution; some dependency is required before search consumes the frame, but not necessarily a host block right there. |
| Renderer D2H frames | `[B,2,1,64,64]` | large, about 8 MiB at B1024 | Required only for host stack/debug/final-observation paths; avoid on hot search path in resident mode. |
| Host stack shift/latest write | `[B,2,4,64,64]` shift plus latest | large, about 32 MiB stack | Host FIFO ownership; no longer needed for hot MCTX if resident stack is authoritative. |
| Resident stack update | `[B,2,4,64,64]` JAX concat | large device op | Current code immediately blocks after concat, so wait may be charged before search. |
| Root batch build | reshaped obs plus sidecars | large if copied, small otherwise | `copy_observation=False` fixed the full obs copy, but host root still validates obs shape and copies sidecars. |
| Search-input H2D | obs + mask or mask only | huge vs tiny | Host mode copies the observation stack back to JAX; resident mode should only transfer the mask/sidecars. |
| Replay/index commit | policy/root/reward/done/final flags | small to medium | Can be chunked; current index rows are cheap in latest rows. |
| Profile timers | JAX/Torch readiness calls | variable | Good for attribution; final throughput rows need a minimal-fence variant too. |

Semantic barriers to preserve:

- selected action before CPU `env.step`;
- done/reward/legal/action-mask sidecars matching the same post-step state as
  the observation;
- terminal final observation before autoreset mutates a row;
- renderer-to-search dependency for the latest frame;
- replay/target commit before learner-facing consumers observe the sample.

## Design Ladder

### 1. Keep root no-copy and make byte accounting explicit

Design: keep `compact_root_copy_observation=False` as the default for this
profile lane, and add/report bytes for observation view, final observation,
legal mask, root ids, and H2D mask/obs.

Expected win: mostly diagnostic; root-copy removal already exposed the resident
stack win. It prevents regressions where a future validation path silently
copies 32 MiB per step again.

Risk: low. Root observation references can be mutated if retained too long.

Validation: root batch metadata must say `observation_copied=false`; sampled
root observation checksum before/after search; compare roots/sec against the
known fast visual/no-copy row (`26,610` class sim16 row).

### 2. Lazy resident stack synchronization

Design: keep the current resident JAX stack update, but remove the immediate
`block_until_ready()` after the FIFO concat. Let MCTX search consume the stack
and become the dependency. Measure a full minimal-fence loop and a fully-fenced
attribution loop.

Measured status: first H100 falsifier is mixed. Sim16 moved from `48.6k` to
`54.7k` roots/sec with explicit resident-stack sync disabled. Sim32 moved from
`36.0k` to `27.9k` roots/sec. Keep this as a profiling switch, not a trusted
speed mode.

Expected win: small at best until repeated. If the explicit resident-stack wait
is pure measurement tax, `resident_stack_update_sec` shrinks and total
roots/sec rises. If the wait just moves into `search_sec` or the residual, keep
the attribution but do not call it an algorithmic win.

Risk: low/medium. Async errors may surface later; timing buckets become less
exclusive.

Validation: same total loop wall denominator; report moved time across
`resident_stack_update_sec`, `h2d_sec`, `search_sec`, and residual. Sample device
stack parity against host stack on warmup/reset-free rows.

### 3. Device observation as the hot MCTX input, host mirror sampled

Design: make resident `[B,2,4,64,64]` the source of truth for MCTX input.
Host `HybridCompactBatch` remains for masks/ids/reward/done/replay validation,
but full host observation stack is updated only on a sampled cadence and for
terminal/final-observation rows.

Expected win: medium/high. It attacks the large bounce:
GPU latest frame -> host stack -> JAX `device_put`. Latest rows already show
resident stack improves sim16 about `1.31x` and sim32 about `1.38x` in the
matched grid; deleting mandatory host mirror work is the next obvious step.

Risk: medium. FIFO order, row/player reshape, warmup zero frames, terminal rows,
and autoreset reset rendering can drift silently.

Validation: sampled parity of resident device stack vs host stack for
`[env_row, player]` order; mandatory parity on terminal/autoreset rows; report
`obs_h2d_bytes=0`, `mask_h2d_bytes>0`, `renderer_device_to_host_sec=0` on hot
steps; keep same active-root denominator.

### 4. Borrow actor render state for actor_count=1

Design: for the single-actor profile shape, pass borrowed env-state render
arrays directly to the renderer instead of copying rows into parent native
render buffers. Existing code has `borrow_single_actor_render_state`; treat it
as a measured falsifier with terminal rows disabled or falling back to copied
snapshot.

Expected win: medium if `actor_render_state_write_sec` is a large leaf. Latest
retests show render-state write around `0.278-0.351s` in no-copy rows, often
larger than strict mechanics.

Risk: medium. Borrowed state can be mutated by autoreset; terminal rows need a
pre-reset snapshot or copied fallback.

Validation: require `actor_count=1`; assert no terminal rows or force fallback;
compare render-state write leaf, production/delta leaves, observation parity,
and total roots/sec against copied native-buffer row.

### 5. Renderer consumes already-compact native render buffers

Design: make `_PersistentJaxPolicyFramebufferRenderer.render` accept a compact
render-state contract directly, bypassing production-to-compact conversion when
the manager already wrote the persistent render keys. Keep current conversion
as oracle/fallback.

Expected win: medium/high. Fast visual adapter reduced production-to-compact to
about `0.054-0.057s` in one row, but the broader state ownership path still
includes render-state write and delta pack.

Risk: medium. The compact contract must match row-major, live-prefix, cursor,
avatar color, bonus, and reset semantics exactly.

Validation: sampled equality of compact state from native buffers vs current
adapter; renderer output parity on first N steps, every K steps, and reset rows;
bucket targets are `renderer_production_to_compact_sec`,
`renderer_persistent_delta_pack_sec`, and `actor_render_state_write_sec`.

### 6. Actor emits compact deltas instead of full render rows

Design: shift the actor/render contract from "copy full render-state rows" to
"emit appended visual trail segments, reset masks, player/bonus compose state".
The persistent renderer applies deltas directly to the device layer.

Expected win: high if delta pack plus row-copy remains hot after design 5. It
also aligns data ownership with the persistent framebuffer: only changed trail
segments move per step.

Risk: high. Deltas must handle wraparound cursors, resets, avatar-color
invalidations, bonus changes, death/warmdown, and first-frame initialization.

Validation: exact replay of full-state renderer over a deterministic trace;
stress cursor wrap/reset/death/bonus cases; compare delta bytes and
`renderer_host_to_device_sec` against current full compact/compose H2D.

### 7. Chunked replay/search output staging

Design: keep actions as per-step readback, but stage visit policy/root value
/search metadata in device or pinned host chunks. Commit replay-index rows at a
chunk boundary, not as part of the action-critical path.

Expected win: small in current rows because replay-index was about `0.3%` in
one matched grid, but useful if observation work shrinks and replay becomes
visible.

Risk: medium. Replay target semantics depend on exact action/reward/done/final
observation alignment across records.

Validation: chunked rows must round-trip through
`build_compact_replay_index_rows_v1_from_search_result`; compare target-row
hashes to current per-step writer; measure replay/index share after designs
2-5.

### 8. Array-native CPU/GPU search boundary

Design: keep CPU mechanics and resident observations, but replace host
`CompactRootBatchV1`/MCTX glue with an array-native service contract:
device obs handle, legal mask `[root,3]`, reward/done/to_play/id sidecars, and
array search outputs. Host object/root validation becomes sampled debug.

Expected win: medium after observation ownership improves. This removes Python
object/root rebuild tax, not the CPU env step.

Risk: high for promotion. It narrows compatibility with stock LightZero and
must prove legal mask, active-root, row/player, root value, and policy target
parity.

Validation: service contract parity against `CompactRootBatchV1` for fixed
seeds; same selected actions/visit distributions within tolerance; sampled host
root batch reconstruction from service arrays.

### 9. GPU-resident tree/search service

Design: move tree state, latent state, visit counts, priors, rewards, and
values into a GPU-resident service. Only selected actions and staged replay
summaries cross to CPU while CPU mechanics remain.

Expected win: high only after observation handoff is no longer the wall. Current
compact MCTX rows show search is single-digit to about `20%`, so this is not the
next sim16 bottleneck.

Risk: very high. Needs a new tree contract, determinism/tolerance envelope,
Dirichlet/temperature semantics, legal masks, and recurrent-model integration.

Validation: no-model tree microbench, then fixed recurrent tensors, then real
model, all compared against current MCTX/CTree contract on small seeds.

### 10. Full GPU/env-resident loop

Design: port CurvyTron mechanics, render deltas, observation stack, search, and
replay staging into one device-resident loop. CPU receives only summaries,
checkpoints, and sampled validation frames.

Expected win: radical ceiling, but poor next-step ROI. Strict mechanics are
only about `8-11%` of `env_step_sec` in the latest rows, or about `5.5-6.5%` of
total wall. Porting mechanics first would attack the wrong slice.

Risk: extreme. This is a rewrite of game semantics, reset/final-observation
ordering, and training data contracts.

Validation: build only after the smaller residency ladder proves the wall moved
from observation/search-input handoff to mechanics/search/replay.

## Recommended Next Prototype

Prototype design 5 first, with a small piece of design 6 if it is simpler:

```text
renderer consumes compact native render state directly
avoid rebuilding a production-style state dict on every hot step
avoid repacking unchanged visual trail data when possible
keep resident device stack as the hot MCTX observation
keep host observation stack sampled/debug or terminal-only
```

Why this first: designs 2/3/4 have already given the useful evidence. Resident
rows beat host rows after root-copy removal. Borrowed render-state removes the
parent visual-state copy. Lazy sync is mixed. In the latest no-sync sim16 row,
the raw GPU draw is only `0.004s` over loop24, while the remaining renderer
leaves are much larger: production-to-compact `0.078s`, delta pack `0.104s`,
H2D `0.067s`, and update `0.005s`. In the sim32 no-sync row those grow to
production-to-compact `0.080s`, delta pack `0.199s`, H2D `0.141s`, and update
`0.009s`. The clean next move is to stop rebuilding and transferring the same
compact render/search input shape every step.

Kill criteria:

- total active roots/sec does not improve at least `10-15%` in the same
  denominator;
- production-to-compact and delta-pack time do not move, or only move into
  unlabeled residual;
- sampled host/device stack parity fails for row/player order or FIFO depth;
- terminal/autoreset final-observation ordering cannot be proven without
  reverting to the copied host path for affected rows.

Promotion criteria:

- hot steps report zero full-observation H2D and zero renderer frame D2H;
- action readback remains the only per-step required semantic CPU/GPU boundary,
  and it is small;
- host stack/root observation copies are sampled/debug or terminal-only;
- same seeds preserve legal masks, selected actions legality, reward/done,
  row/player ids, and compact replay-index contract.
