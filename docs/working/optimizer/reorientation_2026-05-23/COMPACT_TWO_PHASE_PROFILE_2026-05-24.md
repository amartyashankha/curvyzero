# Compact Two-Phase Profile, 2026-05-24

Purpose: record the first matched timing proof after `CompactTorchSearchServiceV1`
grew a real two-phase profile path.

These rows are profile-only. They do not call `train_muzero`, do not touch live
training runs, and are not Coach launch recommendations by themselves.

## Shape

- hardware: H100
- batch roots: B512 and B1024
- actors: 16
- simulations: 8
- measured steps: 200
- warmup steps: 80
- action mode: `scripted_random`
- scalar timestep materialization: off
- observation input: `host_uint8`

## Results

| batch | row | measured sec | steps/sec | probe sec | observation sec | actor sec | H2D sec | model sec | search sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | direct CTree GPU-latent | `33.556` | `6103.25` | `18.777` | `4.974` | `8.058` | `4.363` | `2.995` | `8.443` |
| 512 | compact Torch two-phase | `23.311` | `8785.72` | `10.054` | `4.587` | `7.262` | `1.415` | `0.905` | `7.354` |
| 512 | fixed-shape no-real-search floor | `11.165` | `18343.48` | `0.379` | `3.722` | `6.272` | `0.000` | `0.000` | `0.030` |
| 1024 | direct CTree GPU-latent | `46.776` | `8756.62` | `28.056` | `8.070` | `8.145` | `7.036` | `4.532` | `10.771` |
| 1024 | compact Torch two-phase | `34.019` | `12040.17` | `12.962` | `8.427` | `9.742` | `2.942` | `1.747` | `7.503` |
| 1024 | fixed-shape no-real-search floor | `24.752` | `16548.13` | `0.656` | `9.910` | `10.800` | `0.000` | `0.000` | `0.052` |
| 2048 | direct CTree GPU-latent | `87.807` | `7463.65` | `52.151` | `17.762` | `12.704` | `13.842` | `6.658` | `17.970` |
| 2048 | compact Torch two-phase | `54.619` | `11998.74` | `18.007` | `18.347` | `12.822` | `5.773` | `2.603` | `8.674` |
| 2048 | fixed-shape no-real-search floor | `33.072` | `19816.19` | `1.234` | `15.559` | `13.460` | `0.000` | `0.000` | `0.107` |

## Ratios

| batch | ratio | read |
| ---: | ---: | --- |
| 512 | `1.44x` | compact Torch two-phase over direct CTree |
| 1024 | `1.38x` | compact Torch two-phase over direct CTree |
| 2048 | `1.61x` | compact Torch two-phase over direct CTree |
| 512 | `2.09x` | no-real-search floor over compact Torch two-phase |
| 1024 | `1.37x` | no-real-search floor over compact Torch two-phase |
| 2048 | `1.65x` | no-real-search floor over compact Torch two-phase |
| 512 | `3.01x` | no-real-search floor over direct CTree |
| 1024 | `1.89x` | no-real-search floor over direct CTree |
| 2048 | `2.66x` | no-real-search floor over direct CTree |

## Plain Read

The two-phase compact Torch path is a real profile improvement over direct
CTree in this denominator. The hot path returns selected actions only and
reports zero hot replay-payload bytes.

It is not a `10x` result. At B1024, direct CTree takes `46.8s`, compact Torch
takes `34.0s`, and a no-real-search floor still takes `24.8s`. That means the
new path removes a meaningful chunk, but the remaining time is now mostly
observation movement, actor/env loop time, and fixed compact-batch work plus
some remaining search/model work.

The B2048 follow-up strengthens the same read. Compact Torch is `1.61x` faster
than direct CTree, mostly because direct CTree gets much worse at this root
batch. Compact Torch itself is nearly flat versus B1024 in throughput, around
`12k` steps/sec. The floor reaches `19.8k` steps/sec, so the profile still has
headroom, but the obvious wall has shifted to observation plus actor/env work.

In Amdahl terms: making search output ownership cleaner helps, but it cannot
by itself make the whole loop blazingly fast while the rest of the profile still
costs tens of seconds.

## Suspicious Or Important Details

- `root_observation_copy_bytes` is counted differently across rows. Compact
  Torch now reports the active-root observation staging copy, while direct CTree
  still reports zero for that field. Compare `obs_h2d_bytes`, `H2D sec`, and
  wall time instead of trusting that one field alone.
- The fixed-shape floor is not real MCTS. It is useful only as a lower bound on
  non-search plumbing.
- The profile action stream is controlled with `scripted_random`, so the rows
  are fair for timing, but they do not prove learning quality.
- This still excludes the real trainer denominator: learner updates, replay
  sampling cadence, RND training cadence, checkpointing, and stock LightZero
  trainer object flow.

## Next Moves

1. Run the same matched compact rows at B2048 with the same warmup and action
   stream. If compact Torch gains more than direct CTree as batch grows, keep
   pushing compact ownership; if not, the next wall is fixed env/actor plumbing.
2. Add a compact profile row that flushes replay payloads at realistic commit
   cadence instead of always dropping them in `scripted_random`. This checks
   whether delayed replay stays cheap when it is actually used.
3. Build the smallest trainer-like compact denominator that consumes compact
   replay rows and RND/latest-frame data without scalar `BaseEnvTimestep`
   materialization. Do not claim Coach speed until this denominator exists.

## 2026-05-24 Learner-Edge Patch

Added a profile-only compact learner gate. It sits after the existing compact
sample gate:

```text
CompactReplayIndexRowsV1
-> SourceStateMultiplayerSampleBatchV0
-> tiny Torch learner-like consumer
```

It reports learner-gate calls, updates, sample rows, input bytes, elapsed time,
device, loss, and optional RND-style latest-frame loss. It does not call
`train_muzero`, does not touch live runs, and does not claim LightZero learner
parity. Its job is to stop measuring only collection/search when the next
question is replay/sample/learner-edge cost.

First H100 honesty row:

```text
Shape:
  B512/A16/sim8, 120 measured steps, 80 warmup,
  compact Torch two-phase, search_feedback actions,
  sample gate every 8 commits, sample batch 256,
  learner gate on CUDA, RND-style latest-frame loss enabled.

Result:
  measured_sec                         24.260
  steps_per_sec                         5065.09
  committed_index_rows                122880
  committed_replay_payload_d2h_bytes 3440640
  sample_gate_calls                       15
  sample_gate_sec                       12.432
  learner_gate_calls                       15
  learner_gate_sec                       0.343
  learner_gate_updates                     15
  learner_gate_input_bytes          251746560
```

Plain read: the old `toy_probe` tiny learner consumer is not the wall. It costs
about `1.4%` of this row. The compact sample/materialization gate is the wall,
about `51%` of the measured time. This row strongly redirects the next
optimization from "make the toy learner gate faster" to "stop materializing
compact replay/sample batches through Python copies."

Follow-up fix:

- First, the sample gate sampled after materializing all target rows. Changing
  it to sample compact index rows first reduced sample-gate time from `12.432s`
  to `8.615s`.
- Then the profile gate bypassed the slow target-row list/dict bridge and built
  `SourceStateMultiplayerSampleBatchV0` directly from compact index rows plus
  the two adjacent env steps.

Result after the direct fast sample-batch path:

```text
B512/A16/sim8, same shape:
  measured_sec        11.520
  steps_per_sec    10666.33
  sample_gate_sec      0.096
  learner_gate_sec     0.355
  probe_sec            4.777
  actor_wall_sec       3.401
  observation_sec      2.483
```

Plain read: this was the first genuinely large compact-denominator win in this
round. The row is `2.10x` faster than the first replay/learner honesty row and
`1.79x` faster than the intermediate pre-sampled row. The sample gate is no
longer the wall; search, actor/env, and observation are again the main buckets.

B1024 fast sample-batch row:

```text
B1024/A16/sim8, sample batch 512, otherwise same shape:
  measured_sec        22.771
  steps_per_sec    10792.46
  sample_gate_sec      0.366
  learner_gate_sec     0.476
  probe_sec            8.359
  actor_wall_sec       5.982
  observation_sec      5.556
```

This is about `10%` slower than the earlier B1024 action-only compact row
(`12040.17` steps/sec), while actually flushing replay payloads, sampling
learner-shaped batches, and running the tiny CUDA learner/RND consumer. That is
a much healthier denominator.

## 2026-05-24 Copy-Cleanup Confirmation Row

After removing duplicate observation copies in the compact Torch and compact
batch paths, a warm H100 row was run with direct64/persistent renderer,
`uint8` stack storage, compact Torch search feedback, replay flush, sample
gate, and tiny CUDA learner/RND gate:

```text
B1024/A16/sim8, 120 measured steps, 100 warmup:
  steps/sec                       4769.84
  measured_sec                      51.524
  actor_env_runtime_sec             39.259
  observation_sec                    4.217
  compact_rollout_slab_sec           5.794
  compact_search_service_total_sec   5.368
  compact_search_sec                 3.870
  compact_model_sec                  1.022
  sample_gate_sec                    0.310
  learner_gate_sec                   0.380
  compact_batch_build_sec            0.007
  root_observation_copy_bytes        0
  root_mask_copy_bytes               0
  env_action_checksum_total          0
```

Plain read: the copy cleanup worked. Root observation/mask staging no longer
reports a host gather copy for the all-active contiguous root batch, and compact
batch build time is tiny.

This row is not a clean speed comparison. `search_feedback` selected action
`0` everywhere, so the trajectory is action-confounded and actor/env runtime
dominates the wall. Use it as a correctness/accounting confirmation, not as a
claim that compact Torch got slower than the earlier controlled rows.
