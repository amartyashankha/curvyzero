# Resident Chunk Canary Plan

Date: 2026-05-21

Status: proposed next optimizer lane. Profile-only; no trainer, tournament, or
live-run defaults change.

## Why This Exists

Fresh profiles say pure observation/render work is no longer the whole wall.

At C512/sim8:

```text
L4 real batched obs:   ~1077 steps/s
L4 zero observation:   ~1263 steps/s
H100 real batched obs: ~1378 steps/s
H100 zero observation: ~1608 steps/s
```

So deleting observation entirely buys only about `1.17x`. That means another
renderer-only rewrite cannot deliver the next `5x-10x`.

The likely big win is preserving a large batch through more of the loop.

## Target Shape

Build a canary that measures this profile-only path:

```text
compact CurvyTron state batch
-> batched observation stack
-> batched GPU model/search-like consumer over B*players roots
-> optional scalar LightZero-shaped rows at the edge
```

This is not a trainer. It is a measurement tool for deciding whether a real
architecture change is worth building.

## What It Must Measure

- env/source-state step time;
- observation render/stack time;
- host/device bytes and sync count where practical;
- batched policy/model call time;
- search-like loop time or a clearly labeled synthetic stand-in;
- scalar materialization cost when enabled;
- effective scalar steps/sec and root evaluations/sec.

## What It Must Not Do

- no live training writes;
- no tournament default changes;
- no checkpoint/eval/GIF side effects;
- no hidden CPU fallback;
- no claim that synthetic search equals real MCTS.

## First Useful Rows

Use no-death first so trajectories are long enough:

| row | compute | B | T | consumer | scalar edge |
| --- | --- | ---: | ---: | --- | --- |
| A | L4/T4+40CPU | 512 | 128 | batched model-like GPU consumer | off |
| B | L4/T4+40CPU | 512 | 128 | batched model-like GPU consumer | on |
| C | H100+40CPU | 512 | 128 | batched model-like GPU consumer | off |
| D | H100+40CPU | 512 | 128 | batched model-like GPU consumer | on |

Then repeat at B1024 only if B512 shows scaling headroom.

## Success Criteria

The canary is worth promoting to the next architecture prototype only if:

1. scalar-edge-off throughput is clearly above the current stock-profile zero
   observation ceiling;
2. scalar-edge-on quantifies how much LightZero-shaped materialization costs;
3. the model/search-like consumer keeps GPU utilization meaningfully higher
   than the current stock profile;
4. row/player/action-mask/reset/final-observation contracts remain explicit;
5. results are recorded in this folder before any Coach recommendation.

## Current Best Guess

If this works, the next real implementation is not "make the trainer use GPU
rendering." It is one of:

- central batched actor/search service feeding the stock trainer/replay edge;
- custom collector boundary that stores chunks before LightZero scalarization;
- deeper JAX/Mctx research lane for accelerator-resident search.

If this does not beat the current zero-observation ceiling, the practical next
work is likely CPU/manager/search cleanup inside the stock LightZero path.

## 2026-05-21 Implementation Pass

The smallest honest canary is already present in the profile-only hybrid path,
so do not add another dummy benchmark:

```text
VectorMultiplayerEnv actors
-> compact parent payload merge
-> renderer-backed [B, players, 4, 64, 64] stack
-> HybridBatchedStackProbe before scalarization
-> optional materialize_lightzero_scalar_timestep at the edge
```

Use the Modal boundary entrypoint for GPU rows:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --hybrid-observation-canary \
  --compute gpu-h100 \
  --batch-size 512 \
  --actor-count 16 \
  --steps 100 \
  --warmup-steps 20 \
  --trail-slots 1024 \
  --body-capacity 1024 \
  --dynamic-render-trail-slots \
  --render-surface direct_gray64 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --hybrid-stack-storage-dtype uint8 \
  --hybrid-batched-stack-probe-simulations 8 \
  --hybrid-batched-stack-probe-channels 16 \
  --hybrid-materialize-scalar-timestep false
```

Then repeat the same row with `--hybrid-materialize-scalar-timestep true`.
The off row measures the resident chunk shape before LightZero scalar rows.
The on row prices scalar materialization at the boundary.

For the L4/T4 version of this boundary app, use:

```text
--compute gpu-l4-t4
```

Do not use `gpu-h100-cpu40` or `gpu-l4-t4-cpu40` with this entrypoint. Those
names are train-profile grid aliases, not boundary-profile compute names.

Optional device-latest row:

```bash
  --hybrid-batched-stack-probe-device-latest
```

This row is only valid with
`jax_gpu_persistent_policy_framebuffer_profile`. It is a copy-avoidance canary,
not a production stack contract: the manager still maintains a host stack today.

### Measures

- env step and compact merge;
- renderer render time, device render time, and stack update time;
- pre-scalar batched stack probe H2D, normalize, device work, readback, roots,
  input dtype, and input bytes;
- optional scalar materialization time and materialized timestep count;
- physical rows/sec and scalar root steps/sec.

### Excludes

- stock `train_muzero`;
- real LightZero MCTS;
- replay writes, checkpoint writes, eval, GIF, tournament, or live-run changes;
- subprocess actor IPC;
- a claim that the synthetic probe is algorithmically equivalent to search.

### Promotion Gate

Only promote this into a real architecture prototype if the scalar-off row beats
the C512/sim8 zero-observation ceiling by a material margin while keeping
renderer, stack, probe, and scalar-edge costs separately visible. If it only
moves time between host stack update, H2D, normalize, and readback, keep the
work in the profile lane.

## 2026-05-21 First Sim8 Read

The first sim8 repeat cleared the profile-only gate:

| compute | scalar edge | device-latest | scalar roots/sec | read |
| --- | ---: | ---: | ---: | --- |
| H100 | off | no | `13773.66` | fastest clean resident row |
| H100 | on | no | `6487.59` | scalar materialization roughly halves throughput |
| L4/T4 | off | no | `8970.81` | cheaper GPU still useful |
| L4/T4 | on | no | `4221.22` | same scalar-edge collapse |
| H100 | off | yes | `9762.40` | H2D lower, total slower because host stack remains |

Decision:

```text
The resident canary is worth the next prototype. The next prototype must add
real policy/search/replay-shaped pressure without immediately collapsing back
to scalar LightZero rows.
```

Do not tell Coach this is a training speed number. It is a systems proof that
batch residency can create the class of headroom we were looking for.

## 2026-05-21 Resident Replay/Search Probe Read

The profile-only resident replay/search probe is now implemented. It is still
synthetic, but it is harsher than the first stack-only canary because it writes
to a replay-like device ring, samples from it, and runs policy/search-shaped
GPU work.

Medium rows, B512/A16/sim8, 60 measured steps, 20 warmup steps:

| compute | scalar edge | scalar roots/sec | measured sec | observation sec | resident probe sec | scalarization sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H100 | off | `10980.47` | `5.60` | `2.33` | `0.58` | `0.00` |
| H100 | on | `7620.76` | `8.06` | `2.45` | `0.67` | `1.96` |
| L4/T4 | off | `5839.67` | `10.52` | `3.44` | `2.87` | `0.00` |
| L4/T4 | on | `4133.28` | `14.86` | `3.70` | `2.86` | `3.68` |

Read:

- H100 is about `1.88x` faster than L4/T4 when the synthetic probe is heavy
  enough.
- Scalar materialization is expensive, but not a total collapse in this
  profile-only row: H100 keeps about `69%` of scalar-off throughput and L4/T4
  keeps about `71%`.
- The row creates `61,440` scalar timesteps when scalar edge is on. That is
  exactly why scalar materialization should remain an edge cost, not the hot
  loop.
- This still is not training. The next prototype must use a real or closer
  LightZero policy/search consumer.

## Promotion Guardrails

Do not promote resident batching into Coach launch advice until these gates are
reported in the same denominator as the real trainer:

1. matched no-RND rows: CPU oracle, resident/batched candidate, and zero
   observation;
2. matched `rnd_meter_v0` rows with update cadence, predictor/target hash
   checks, reward-unchanged checks, and RND timing;
3. normal-death/autoreset row with terminal `final_observation`, partial row
   masks, and reset semantics visible;
4. long-survival/no-death row so trail/render/stack walls are not hidden by
   short dead games;
5. real or realistic policy/search pressure row. Synthetic stack probes must
   stay labeled synthetic until this passes.

Positive RND reward remains blocked for recommendation until normalization,
resume/checkpoint state, support metadata, seed robustness, and extrinsic
reward quality are settled.

## Implemented Prototype: Resident Replay/Search Probe

This patch is profile-only and Modal-only. It is not wired into trainers,
tournaments, eval, checkpoints, or defaults.

The `_JaxHybridResidentChunkProbe` style consumer currently keeps observation
and action-mask arrays on device:

```text
device_stack:        [B, P, 4, 64, 64] uint8
replay_obs:          [capacity, 4, 64, 64] uint8
replay_action_mask:  [capacity, ACTION_COUNT] bool
write_cursor:        int32
valid_count:         int32
```

Reward/done/row/player metadata are still future extensions. Do not claim the
probe is replay-complete yet.

First mode:

```text
hybrid_observation_canary=true
hybrid_stack_storage_dtype=uint8
hybrid_materialize_scalar_timestep=false
hybrid_batched_stack_probe_device_latest=false
```

Why no device-latest first: the current device-latest row reduced H2D but was
slower overall because the host stack is still maintained. First prove the
resident replay/search pressure is valuable with the existing host-fed stack;
then remove the host stack in a separate gate.

Suggested Modal-only flags:

```text
hybrid_resident_chunk_probe: bool = false
hybrid_resident_replay_steps: int = 64
hybrid_resident_sample_batch_size: int = 256
hybrid_resident_replay_train_steps: int = 1
hybrid_resident_readback_checksum: bool = true
```

First telemetry fields can live inside `batched_stack_probe_last_telemetry`:

```text
resident_stack_h2d_sec
resident_action_mask_h2d_sec
resident_metadata_h2d_sec
resident_replay_write_sec
resident_replay_sample_sec
resident_policy_search_sec
resident_replay_train_sec
resident_readback_sec
resident_total_sec
resident_host_to_device_bytes
resident_replay_capacity
resident_replay_valid_count
resident_replay_write_count
resident_sample_batch_size
resident_model_eval_count
```

Historical success criteria for this probe:

- all profile-only flags remain true/false as expected;
- `materialize_scalar_timestep=false`;
- `materialized_timestep_count == 0`;
- `last_flat_obs_shape == [0, 4, 64, 64]`;
- replay writes exactly one row per root;
- H100 sim8 remains at least `10k` roots/sec or at least `75%` of the current
  scalar-off canary, and stays clearly above scalar-on `~6.5k`;
- resident policy/search plus replay-train timing is non-trivial, so the probe
  is not optimized away.

Failure criteria:

- throughput collapses to scalar-on range;
- scalar materialization appears in timings;
- host-to-device bytes become accidental float32-stack sized;
- compile appears in measured steps;
- device-latest claims a win while full host stacks are still transferred.

Next success criteria:

- replace or supplement synthetic convolution loops with actual LightZero
  policy/search-shaped work;
- keep the resident batch alive through that real consumer before scalarizing;
- report whether MCTS/search itself forces CPU NumPy residency.
