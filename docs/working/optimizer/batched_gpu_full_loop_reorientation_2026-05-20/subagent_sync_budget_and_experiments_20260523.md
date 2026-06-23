# Sync Budget And Data Movement Experiments, 2026-05-23

Scope: docs-only optimizer critique pass. I read the current optimizer docs and
profile code. I did not touch live Coach runs, checkpoints, evals, GIFs,
tournaments, Modal volumes, or source code.

Requested shape:

```text
B256/B512 env rows
P=2 players
R=B*P active search roots when no-death/full-active
actor_count=16
frame stack [4,64,64]
action_count=3
sim16/sim32
```

Main files read:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/training/compact_policy_row_bridge.py`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_current_code_dataflow_sync_map_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_host_overhead_sync_audit_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_gpu_sync_model_20260522.md`

## Plain Read

The byte sizes are not scary for actions, masks, visit policies, or root
values. Those are KiB-scale payloads.

The scary part is synchronization shape:

```text
CPU tree/control step
-> small CPU-to-GPU action payload
-> GPU recurrent model
-> small GPU-to-CPU model output payload
-> Python listify
-> CPU tree backprop
```

If that happens once per simulation, the payload bytes can be tiny and still be
structurally bad. At `sim16`, that is 16 host/device round trips inside one env
step. At `sim32`, it is 32. This is the key reason the current LightZero CTree
GPU-latent path can still be slow even though the model runs on the GPU.

The healthy shape is:

```text
one batched root input per step
-> search loop stays device-side or in one compiled/service boundary
-> one small selected-action readback before CPU env step
-> one small replay payload readback before replay commit
```

## Payload Sizes

Assume all roots are active. If deaths or zero legal masks drop roots, these
numbers scale down with active `R`.

| Payload | Shape | B256, R512 | B512, R1024 | Read |
| --- | --- | ---: | ---: | --- |
| Joint action | `[B,2] int16` | `1 KiB` | `2 KiB` | CPU-owned action for env step. |
| Selected action | `[R] int32` | `2 KiB` | `4 KiB` | GPU-to-CPU. Mandatory before CPU env step. If int64, double it; still tiny. |
| Legal/invalid mask | `[R,3] bool` | `1.5 KiB` | `3 KiB` | CPU-to-GPU today. Bytes are harmless; waits can still hurt. |
| Visit policy | `[R,3] float32` | `6 KiB` | `12 KiB` | GPU-to-CPU for replay target. |
| Root value | `[R] float32` | `2 KiB` | `4 KiB` | GPU-to-CPU for replay target. |
| Latest frame | `[B,2,1,64,64] uint8` | `2 MiB` | `4 MiB` | Should stay GPU-resident in the hot loop. |
| Policy stack | `[B,2,4,64,64] uint8` | `8 MiB` | `16 MiB` | Search input. Avoid hot-loop copies if possible. |
| Policy stack | `[B,2,4,64,64] float32` | `32 MiB` | `64 MiB` | Avoid host-side float32 stack in the hot loop. |
| Root observation copy | `[R,4,64,64] uint8` | `8 MiB` | `16 MiB` | Current best compact rows avoid this copy. |
| Per-sim recurrent output | `[R,5] float32` for reward/value/logits | `10 KiB/sim` | `20 KiB/sim` | Small bytes, bad if copied and synchronized every sim. |
| Per-sim leaf action | `[R] int64` worst case | `4 KiB/sim` | `8 KiB/sim` | Small bytes, bad if copied and synchronized every sim. |

Per-sim totals for the current CPU CTree style are still small by bytes:

| Shape | sim16 leaf action H2D | sim16 model output D2H | sim32 leaf action H2D | sim32 model output D2H |
| --- | ---: | ---: | ---: | ---: |
| B256/R512 | `64 KiB` | `160 KiB` | `128 KiB` | `320 KiB` |
| B512/R1024 | `128 KiB` | `320 KiB` | `256 KiB` | `640 KiB` |

These byte totals should be cheap if batched once. They are not cheap if each
simulation forces a device wait, CPU conversion, Python list conversion, and
CPU tree call.

## Where We Pay

### Per Step

Paid once per environment/search step:

1. CPU actor/env step over `B` rows.
2. Actor merge or native compact buffer writes.
3. Production/render state to compact state.
4. Delta packing for the persistent renderer.
5. Host-to-device render/update payloads.
6. Device render/update and resident stack update.
7. Compact root batch sidecars.
8. Root input transfer or resident stack readiness.
9. Search.
10. Selected-action readback.
11. Replay target payload and compact replay rows if replay-valid mode is on.

For `actor_count=16`, the total tensor sizes do not change. The slice size does:

```text
B256: 16 rows per actor
B512: 32 rows per actor
```

This means action payloads per actor are tiny. The risk with many actors is
process/message/merge overhead and copying large render state out of each actor.
Native compact buffers are the right direction because they avoid large returned
payload objects.

### Per Root

Paid proportional to active roots `R`:

1. Initial model inference over root observations.
2. Policy logits/value arrays.
3. Legal-action filtering and root setup.
4. Root noise and prior preparation.
5. Selected action, visit policy, root value output arrays.
6. Compact result validation and replay row construction.

Per-root work is fine when it stays in arrays. It becomes bad when it turns into
Python lists or per-root objects before search/replay truly needs that shape.

### Per Simulation

This is the most dangerous layer.

The current direct LightZero CTree GPU-latent path still has a per-simulation
host/device loop:

1. `batch_traverse(...)` runs CPU CTree traversal.
2. Leaf latent indices are used to gather GPU latent states.
3. `last_actions` from CTree are converted to a tensor on GPU.
4. The code synchronizes after moving those actions.
5. `model.recurrent_inference(...)` runs on GPU.
6. The code synchronizes after recurrent inference.
7. Reward, value, and policy logits are copied back to CPU.
8. Those arrays are converted to Python lists.
9. `batch_backpropagate(...)` runs CPU CTree backprop.

That means `sim32` doubles the number of host-visible control turns compared
with `sim16`. This is structurally bad even though each transfer is only KiB.

The MCTX/JAX sidecar shape is healthier because the simulation loop is inside a
single JAX search call. It still reads selected actions back for the CPU env,
but it does not need a CPU round trip for every simulation.

## Current Measured Anchors

Fresh same-shape service comparator from `experiment_log.md`:

```text
H100, B512, actor_count=16, 60 measured steps, 15 warmup,
frame stack [4,64,64], action_count=3, sim16.

mock_search_service:       17,711.9 steps/sec
service_tax_probe:         12,461.6 steps/sec
direct_ctree_gpu_latent:    7,155.7 steps/sec
```

Direct CTree timing on that row:

```text
measured total:                              8.586s over 60 measured steps
LightZero MCTS arrays boundary total:        5.048s
LightZero MCTS arrays boundary search:       3.941s
LightZero model total:                       1.290s
LightZero non-model direct boundary:         3.758s
CTree traverse + backprop:                   1.037s
root prepare:                                0.494s
observation / renderer stack update:         1.263s
actor step wall:                             1.550s
compact replay proof:                        0.174s
```

Approximate per measured step:

```text
total:                         143 ms/step
MCTS arrays boundary:           84 ms/step
MCTS search:                    66 ms/step
model total:                    22 ms/step
non-model direct boundary:      63 ms/step
CTree traverse + backprop:      17 ms/step
root prepare:                    8 ms/step
observation/update:             21 ms/step
actor step wall:                26 ms/step
```

Plain meaning: at this shape, the big wall is not the selected-action readback,
the mask transfer, or the replay row. The big wall is the search boundary plus
CPU/env/observation handoff around it. In the direct CTree route, the per-sim
CPU/GPU loop is a prime suspect.

## Harmless Syncs

These syncs are acceptable if they happen once per step or at commit boundaries:

1. **Selected action GPU-to-CPU before CPU env step.**
   The CPU env cannot advance without actions. Payload is `2-4 KiB` for
   B256/B512 if int32.

2. **Invalid mask CPU-to-GPU once per step.**
   Payload is `1.5-3 KiB`. This is fine if it is one batched copy and does not
   force unrelated queued work to finish.

3. **Visit policy and root value GPU-to-CPU for replay commit.**
   Payload is `8 KiB` at B256 and `16 KiB` at B512 combined. This is fine once
   per step after search, especially after the direct root-value extraction fix.

4. **Terminal/final-observation snapshot.**
   This can be slower because it is rare and correctness-critical.

5. **Warmup/end-of-run profiling barriers.**
   Fine when clearly labeled. They should not be used to claim production hot
   path cost.

## Structurally Bad Syncs

These are bad even when payload bytes look small:

1. **Per-simulation leaf action H2D plus sync.**
   This happens inside the direct CTree GPU-latent loop. It is repeated
   `num_simulations` times.

2. **Per-simulation recurrent output D2H plus sync.**
   The `[R,5] float32` output is small, but it feeds CPU CTree backprop each
   simulation. This serializes GPU model work with CPU tree work.

3. **Per-simulation Python listification.**
   Reward/value/policy logits are converted to Python lists before CTree
   backprop. That is array work becoming Python object work inside the sim loop.

4. **Full stack host movement per step.**
   B512 stack is `16 MiB` as uint8 and `64 MiB` as float32. This should stay
   resident or be copied only at sampled validation edges.

5. **Full latest-frame D2H every step.**
   B512 latest frame is `4 MiB`. It is fine for GIFs, parity samples, and
   terminal/final-observation logic; it is bad if done just to feed search.

6. **Many small blocking `device_put`s.**
   Renderer delta/compose state is made of multiple arrays. Bytes may be modest,
   but per-array blocking can cost driver/sync overhead. Batch or defer waits
   where possible.

7. **Root-value fallback through full search summary.**
   Current docs say direct root-node extraction fixed this. If fallback summary
   reappears in hot rows, treat it as a regression.

## Five Low-Risk Experiments

These should be profile-only. Do not touch live Coach runs.

### 1. Same-Denominator Service Comparator Grid

Run the existing B512/A16 comparator and add B256:

```text
B256/A16 sim16
B256/A16 sim32
B512/A16 sim16
B512/A16 sim32
```

For each shape, compare:

```text
mock_search_service
service_tax_probe
direct_ctree_gpu_latent
```

Record:

```text
steps/sec
measured_sec
lightzero_mcts_arrays_boundary_total_sec
lightzero_mcts_arrays_boundary_search_sec
lightzero_consumer_model_total_sec
lightzero_consumer_direct_boundary_non_model_sec
ctree traverse/backprop
root_prepare_sec
observation_sec
actor_step_wall_sec
```

Expected validation: fixed per-step buckets should roughly double from B256 to
B512; per-sim buckets should roughly double from sim16 to sim32. If direct
CTree gets much worse while service-tax stays closer to linear, the CPU CTree
per-sim boundary is the target.

### 2. Precomputed-Recurrent CTree Ablation

Compare on B256/B512 and sim16/sim32:

```text
direct_ctree_gpu_latent
direct_ctree_gpu_latent_precomputed_recurrent
```

The precomputed-recurrent lane removes real recurrent model compute but keeps
much of the CPU CTree traversal/backprop, model-output copy/listify shape, and
per-sim control structure.

Expected validation:

- If precomputed is still slow, CTree/list/control/sync is the wall.
- If precomputed becomes much faster, recurrent inference is the wall.
- If model output D2H/listify remains large even with fake recurrent payloads,
  we have direct evidence that per-sim host materialization is expensive.

### 3. Tiny Transfer Sync Microbench

Run a toy Modal H100 microbench with the same payload sizes but no env and no
model:

```text
R512 and R1024
S16 and S32
case A: per-sim copy [R] actions H2D, sync, copy [R,5] output D2H, sync
case B: one batched copy [S,R] actions and [S,R,5] outputs, one sync
case C: all-device loop, one final selected-action readback
```

Expected validation: if case A is much slower than B/C despite tiny bytes, the
problem is synchronization cadence, not bandwidth. This directly prices the
bad CTree-shaped boundary.

### 4. Input Residency And Stack-Copy Grid

Use existing input modes where possible:

```text
host_uint8_pinned
host_float32
resident/reuse ceiling, clearly labeled stale if stale
resident visual stack / no root observation copy in the MCTX compact loop
```

Run B256/B512, sim16/sim32. Record:

```text
tensor_prepare_sec
host_to_device_sec
host_to_device_bytes
normalize_sec
resident_first_fill_sec
resident_reused
root_observation_copied
```

Expected validation: B512 stack copies should show about `16 MiB` uint8 or
`64 MiB` float32 transfer exposure. If H2D is small and the wall persists, stop
blaming stack bandwidth. If H2D or tensor prepare grows sharply with B, keeping
the stack resident becomes more urgent.

### 5. Search Output Payload And Replay Commit Check

Run replay-valid full payload versus action-only diagnostic on B256/B512 and
sim16/sim32, after the direct root-value extraction fix:

```text
full replay-valid payload
action-only diagnostic
deferred payload diagnostic, if already available
```

Record:

```text
d2h_action_sec if available
d2h_visit_policy_sec or total d2h_sec
root_value_extract_sec
root_value_source
replay_index_sec
output_readback_bytes
```

Expected validation: visit policy and root values should be small. If full
payload is close to action-only, replay payload is not the next target. If full
payload regresses badly, check for accidental root summary fallback or full
object materialization.

## Decision Rule

Use these experiments to separate three cases:

1. **Per-sim CTree boundary dominates.**
   Move toward a device-side search implementation or compact search service
   that does not bounce recurrent outputs through CPU every sim.

2. **Input residency dominates.**
   Keep policy stacks device-resident, copy only masks/actions/targets, and
   avoid full root-observation/float32 stack movement.

3. **Actor/env/observation handoff dominates.**
   Focus on compact state ownership, fewer render-state copies, batched delta
   payloads, and sampled validation instead of per-step public packaging.

The likely current answer is a mix of 1 and 3. At sim16, observation/env handoff
is still large. At sim32 and above, the search boundary becomes large enough
that the per-simulation CPU/GPU CTree loop is the more aggressive optimizer
target.
