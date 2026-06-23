# Full Iteration Dataflow Critique, 2026-05-23

Status: docs-only critique note. I inspected the current optimizer docs and the
profile source. I did not touch live Coach training runs, checkpoints, evals,
GIFs, tournaments, Modal volumes, or source code.

## Scope

This note is about the profile-only compact GPU/LightZero lane in:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
src/curvyzero/training/source_state_hybrid_observation_profile.py
src/curvyzero/training/compact_policy_row_bridge.py
src/curvyzero/training/compact_search_service.py
```

It does not call `lzero.entry.train_muzero`. It is not live Coach training.

Important distinction:

```text
LightZero service-comparator lane:
  CPU env -> JAX renderer -> host uint8 stack -> Torch/LightZero probe

Older compact MCTX resident lane:
  CPU env -> JAX renderer -> resident JAX stack -> MCTX search
```

Several docs use both currencies. Do not compare their roots/sec or timing
shares directly.

## Current Iteration Loop

In the current hybrid profile harness, each measured iteration is:

```text
previous action source
-> CPU CurvyTron step
-> CPU compact sidecars
-> JAX persistent renderer
-> host or device observation stack
-> compact LightZero probe/search
-> compact search arrays
-> optional replay proof
-> next action source
```

### 1. Action Input

What is created:

- `joint_action[B,2] int16` on CPU.
- If `hybrid_compact_service_replay_proof` is off, actions are random.
- If replay proof is on and a previous search result exists, actions come from
  the previous compact search arrays.

Size:

- Small. At `B=512`, about `2 KiB`; at `B=1024`, about `4 KiB`.

Critical read:

- A row without replay proof is not a closed policy loop. It profiles search
  as a consumer of observations, but the search result is not driving the next
  env step.

### 2. CPU Env Step

What is created:

- `VectorMultiplayerEnv` advances CurvyTron mechanics on CPU.
- The actor/manager writes reward, done, episode step, elapsed time, round id,
  alive mask, legal action mask, terminal rows, autoreset rows, and joint
  action sidecars.
- With `native_actor_buffer=True`, actors write into parent compact buffers.

Residency:

- Mechanics state is CPU/NumPy.
- Rewards, dones, masks, ids, and metadata are CPU arrays.

Size:

- Scalar sidecars are small.
- Render/trail state can be large if copied: trail position/radius/owner/active
  arrays are tens of MiB at large `B` and `body_capacity`.

Sync:

- If search output drives the next step, selected actions must be CPU-visible
  before this step can run. That sync is semantically required while env
  mechanics are CPU-owned.

### 3. Render/Search-Input Preparation

What is created:

- The persistent JAX renderer receives CPU production/render state.
- It builds a compact render state on CPU.
- It computes CPU delta state from previous cursor/owner state.
- It sends delta and compose state to JAX.
- It updates the persistent GPU framebuffer/layer.
- It composes the latest policy frame.

Residency:

- CPU: production state, compact state, delta state, compose sidecars.
- GPU/JAX: persistent layer/framebuffer and `last_output_device`.

Size:

- Latest frame `[B,2,1,64,64] uint8`: about `4 MiB` at `B=512`, `8 MiB` at
  `B=1024`.
- Full stack `[B,2,4,64,64] uint8`: about `16 MiB` at `B=512`, `32 MiB` at
  `B=1024`.

Sync:

- By default, H2D delta/compose copies block.
- Persistent update blocks.
- Compose/draw blocks unless async device-only profiling is enabled.
- If host-stack mode is active, the latest frame is copied GPU -> CPU.

Critical read:

- Raw GPU draw is small. The larger cost is the envelope around the draw:
  CPU production-to-compact, delta pack, H2D/update waits, stack/root-input
  ownership, and packaging.

### 4. Observation Stack And Compact Batch

What is created:

- Host FIFO stack `_zero_stack[B,2,4,64,64]`.
- `HybridCompactBatch` with observation, action mask, reward, done,
  `policy_env_id`, row/player ids, target reward, done roots, active roots,
  final observation sidecars, terminal/autoreset rows, alive, and joint action.

Residency:

- In the current LightZero service-comparator command shape, the probe starts
  from the host uint8 stack.
- Separate device-only/resident modes exist, but they are different profile
  currencies.

Size:

- Observation stack is the large object.
- Masks and sidecars are small.

Sync:

- Host-stack mode already paid GPU -> CPU frame readback before the stack can
  be consumed by the Torch/LightZero probe.

Critical read:

- `CompactRootBatchV1(copy_observation=False)` can avoid copying a full root
  observation, but the host stack may still exist and be the source input for
  the LightZero probe.

### 5. LightZero Probe/Search

There are three comparator families.

`mock_search_service`:

- Real env/render/stack.
- Real Torch model initial inference.
- Fake search: selected action and visit policy come from masked policy logits.
- No recurrent rollout, no CTree.

`service_tax_probe`:

- Real env/render/stack.
- Real initial inference.
- Real recurrent inference once per requested simulation.
- Fake tree/search update.
- No CTree traverse/backprop.

`direct_ctree_gpu_latent`:

- Real env/render/stack.
- Real MuZero initial inference.
- Real LightZero CTree roots.
- Real CTree traverse/backprop.
- Real recurrent inference per simulation.
- GPU latent pool is kept on device, but CTree still consumes CPU/list-shaped
  reward, value, and policy payloads.

Data movement in direct CTree:

```text
host uint8 stack
-> Torch CUDA tensor
-> normalized float tensor
-> initial_inference on GPU
-> root values/logits to CPU for CTree root prep
-> for each simulation:
     CPU CTree batch_traverse
     CPU last actions -> GPU
     GPU recurrent_inference
     GPU reward/value/policy -> CPU
     CPU listify
     CPU CTree batch_backpropagate
-> final action/visit/value compact arrays on CPU
```

Size:

- Stack input is large by bytes.
- Per-root action/visit/value outputs are small:
  selected actions `[R]`, visit policy `[R,3]`, root value `[R]`.
- Per-simulation model outputs are small by bytes but expensive because they
  force many ordering points and Python/list work.

Sync:

- H2D stack transfer and normalization synchronize.
- Initial inference synchronizes.
- Direct CTree synchronizes every recurrent call.
- Recurrent outputs are copied GPU -> CPU every simulation.
- Final compact arrays are CPU-visible.

Critical read:

- H100 utilization can stay low because this is a ping-pong workload: GPU
  model calls are repeatedly interrupted by CPU CTree/list work.

### 6. Compact Search Result And Replay Proof

What is created:

- `CompactRootBatchV1`: root observation view/copy, legal mask, active mask,
  to-play, row/player ids, policy ids, rewards, done flags, final sidecars.
- `CompactSearchResultV1`: active root index, row/player ids, policy ids,
  selected action, visit policy, root value, optional raw counts/predictions.
- `CompactReplayIndexRowsV1`: index-only replay rows without copying full
  observations.

Residency:

- These objects are CPU/NumPy validation and replay-contract objects.

Size:

- Small compared with visual stacks.
- `CompactRootBatchV1.observation` can be large if copied; the no-copy path is
  important.

Sync:

- Replay proof consumes CPU-visible selected action, visit policy, and root
  value.
- It also needs current-step reward/done/final sidecars from the next env step.

Critical read:

- The new `CompactSearchServiceV1` protocol and adapters prove a cleaner
  interface shape, but the current hot path still mostly calls existing probes.
  The API exists; it is not yet a full ownership rewrite.

## Latest Timing Read

Fresh same-shape service comparator, H100, `B=512`, `actor_count=16`,
`steps=60`, `warmup=15`, compact replay proof on:

```text
mock_search_service:       17,711.9 steps/sec
service_tax_probe:         12,461.6 steps/sec
direct_ctree_gpu_latent:    7,155.7 steps/sec
```

Direct CTree row timing:

```text
measured_sec:                              8.586
batched_stack_probe_wall_sec:             5.423
lightzero_mcts_arrays_boundary_total_sec: 5.048
lightzero_mcts_arrays_boundary_search_sec:3.941
model_total_sec:                          1.290
direct_boundary_non_model_sec:            3.758
ctree traverse + backprop:                1.037
root_prepare_sec:                         0.494
observation / renderer stack update:      1.263
actor_step_wall_sec:                      1.550
compact replay proof:                     0.174
raw device render:                        0.013
renderer H2D:                             0.252
```

Plain read:

- Search/boundary work dominates this row.
- Observation/render handoff is still visible, but not the whole wall.
- Replay proof is small here.
- Raw draw is tiny.
- The mock/service-tax gap says a compact service boundary has headroom, but it
  does not prove a 10x win by itself.

## Large Versus Small Data

At `B=512`, `P=2`, `R=1024`:

| Data | Shape | Rough size | Direction |
| --- | --- | ---: | --- |
| joint action | `[B,2] int16` | `2 KiB` | CPU |
| selected action | `[R] int32/int16` | `2-4 KiB` | GPU/search -> CPU |
| legal mask | `[R,3] bool` | `3 KiB` | CPU -> GPU/search |
| visit policy | `[R,3] float32` | `12 KiB` | GPU/search -> CPU |
| root value | `[R] float32` | `4 KiB` | GPU/search -> CPU |
| latest frame | `[B,2,1,64,64] uint8` | `4 MiB` | GPU, sometimes D2H |
| uint8 stack | `[B,2,4,64,64] uint8` | `16 MiB` | host or GPU |
| float32 stack | same | `64 MiB` | avoid in hot path |
| root observation copy | `[R,4,64,64] uint8` | `16 MiB` | avoid if possible |
| visual trail position | `[B,body,2] float32` | body-dependent, large | CPU |

Rule:

- Large-by-bytes: visual stacks, latest frames, root observation copies,
  render/trail state.
- Small-by-bytes but sync-sensitive: actions, masks, visits, values, per-sim
  recurrent outputs.
- Large-by-object-count: Python timesteps, CTree roots/lists, public collect
  outputs, scalar replay objects.

## Sync Points

Required today:

- Selected action must be CPU-visible before CPU env step.
- Current observation, legal mask, row ids, player ids, reward, and done must
  describe the same state before search/replay validation.
- Terminal final observation must be captured before autoreset.
- Replay rows must not become sample-visible until action, visit policy, root
  value, reward, done, final masks, row ids, and player ids are attached to the
  same record.

Likely optimizable:

- Full latest-frame D2H when search can consume a device frame.
- Host stack maintenance when no scalar LightZero timestep is needed.
- Full root observation copies in compact root batches.
- Many small H2D copies for masks/deltas if they create queue-wide waits.
- Per-simulation recurrent output D2H/listify in direct CTree.
- Public LightZero output materialization in the profile hot path.

## Where Our Understanding Could Be Wrong

1. **Currency drift.** Some notes discuss MCTX resident roots/sec, while the
   current requested source file is the LightZero service-comparator profile.
   Those are not the same loop.

2. **Closed-loop drift.** A profile row without compact replay proof uses
   random actions. It is not measuring policy self-play dynamics.

3. **Timer nesting.** `env_step_sec` often includes observation preparation and
   search-input handoff. Nested renderer/mechanics timers should not be summed
   as exclusive buckets.

4. **GPU residency overclaim.** The persistent renderer keeps a GPU
   framebuffer/latest frame, but the current LightZero comparator often returns
   frames to a host stack and then sends the stack to Torch. That is not a fully
   resident GPU observation path.

5. **Service API overclaim.** `CompactSearchServiceV1` exists and has adapters,
   but the profile path is not yet fully owned by that service contract. It can
   still double-run or side-channel if wired carelessly.

6. **Mock mode overclaim.** `mock_search_service` and `service_tax_probe` are
   useful ceilings/falsifiers. They are not MCTS and should not be quoted as
   training speed.

7. **No-death bias.** No-death profiles stress long trajectories and reduce
   terminal churn. They can underprice terminal/final-observation/autoreset
   correctness costs.

8. **RND/learner gap.** This lane does not yet price the full trainer with RND,
   replay sampling, target building, and learner updates. A Coach-facing claim
   needs that denominator separately.

9. **Low GPU utilization interpretation.** Low utilization does not prove the
   GPU is weak or idle for a simple reason; the CPU/GPU ping-pong pattern can
   keep both sides underfed.

10. **Async timing labels.** Removing `block_until_ready()` can move wait time
    into the next consumer instead of deleting it. Judge total wall, not a
    relocated bucket.

## Critique Recommendation

The next dataflow target should be one compact owner for:

```text
CompactRootBatchV1
-> CompactSearchServiceV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
```

But the speed work only matters if it removes real ownership boundaries:

- avoid host visual stack as the hot source if search can consume device data;
- keep root observations no-copy or resident;
- reduce per-simulation CTree CPU/list/readback ping-pong;
- keep scalar LightZero objects at validation/debug edges;
- prove replay/RND/player/final-observation parity before trainer-facing use.

The highest-risk misunderstanding is thinking "the renderer is on GPU, so the
loop is GPU-resident." In the current LightZero profile lane, that is not yet
true.
