# Full Iteration Status

Date: 2026-05-10

Status: plain answer for what has and has not been measured as a full training
iteration.

## Short Answer

Later 2026-05-10 correction: CurvyTron visual survival has now crossed into
real installed LightZero trainer calls in the Coach/training lane, including a
profile-sized frozen-checkpoint opponent run with one learner train call. Treat
the older no-train profile language below as pre-trainer context. Optimizer
still cannot infer policy quality from those runs; Coach owns eval and
checkpoint claims.

The primary target is non-ALE visual LightZero-style stacked frames. The current
visual profiling target is only `debug_visual_tensor` /
`curvyzero_debug_occupancy_gray64/v0`: raw `uint8[1,64,64]` CHW occupancy smoke
frames, optionally normalized to `float32[1,64,64]` CHW for a LightZero-facing
payload and wrapper-stacked to `float32[4,64,64]` for the current profile. It is
not source-faithful visual truth. A bounded installed-runtime profile now proves
CurvyTron debug visual rows can run through LightZero conv MuZero
eval-mode MCTS/search, replay-row construction, sampled `[B,4,64,64]` batch,
and no-step `learn_mode.forward` loss. That no-train profile still does not
call `train_muzero`, `collector.collect`, `MuZeroGameBuffer`, or any optimizer
step; later trainer artifacts are separate Coach/training evidence. The current
CurvyTron scalar
reports are sidecar actor/multiplayer-shaped chunks: source env snapshots,
adapter, `[B,2,106]` ray observations, fake policy/search, action scatter, env
step, and replay chunk write. They do not include a real learner
sample/update/checkpoint cycle and they are not the coach-facing optimizer
target.

Historical context only: LightZero official/control Pong had a real bounded
training-loop profile. It ran `lzero.entry.train_muzero` inside one Modal
training container and stopped after `5` `BaseLearner.train` calls. DI-engine
subprocess env workers, policy, replay buffer, collector, evaluator, learner,
logs, checkpoints, and artifact scan all lived inside that function. That slice
is useful only as archived shape evidence. The active optimizer lane is now
CurvyTron-only.

## Archived LightZero Pong Slice

L4/T4 profile, stopped after `5` learner train calls:

| Bucket | Seconds |
| --- | ---: |
| remote elapsed | `266.1` |
| `train_muzero` wall | `251.6` |
| evaluator eval | `101.4` |
| collector collect/env/MCTS | `140.8` |
| replay sample/target | `0.59` |
| learner train, total | `1.62` |
| checkpoint save | `0.20` |

Counts:

- collector calls: `1`
- evaluator calls: `1`
- env steps collected: `7586`
- replay sample calls: `5`
- learner train calls: `5`
- game segments collected/pushed: `2`

H100 profile, same `5` learner-call cap:

| Bucket | Seconds |
| --- | ---: |
| remote elapsed | `224.0` |
| `train_muzero` wall | `205.5` |
| evaluator eval | `87.4` |
| collector collect/env/MCTS | `98.6` |
| replay sample/target | `0.48` |
| learner train, total | `1.24` |

Read: current LightZero Pong training is synchronous and dominated by
evaluator plus collector/env/MCTS wall time. The actual gradient update calls
are a tiny fraction of this measured slice.

## CurvyTron Scalar-Ray Sidecar Chunk

After the dense batch ray patch, a current source-backed scalar-ray sidecar
chunk at `B=32,T=64` measured:

| Bucket | Seconds |
| --- | ---: |
| loop elapsed | `0.466` |
| source adapter | `0.233` |
| observation packing | `0.182` |
| ray cast | `0.097` |
| env step | `0.045` |
| replay write | `0.0018` |
| throughput | `4392 env row-steps/s` |

This is useful sidecar actor-loop speed evidence. It is not a full training
iteration because there is no real learner sample/update/checkpoint in the
loop, and it is not the primary visual CurvyTron training path.

Latest local controlled-trail scalar-ray refresh, with nonempty source body
circles and no observation phase sub-timers:

| Shape | Loop | Source adapter | Observation | Env step | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=16,T=64` | `0.297s` | `0.138s` | `0.124s` | `0.028s` | `3448/s` |
| `B=32,T=64` | `0.555s` | `0.279s` | `0.217s` | `0.050s` | `3692/s` |
| `B=128,T=16` | `0.354s` | `0.148s` | `0.150s` | `0.052s` | `5786/s` |

Artifacts:
`/private/tmp/curvy-optimizer-source-controlled-b16-t64-notimers/profile_report.json`,
`/private/tmp/curvy-optimizer-source-controlled-b32-t64-notimers/profile_report.json`,
and
`/private/tmp/curvy-optimizer-source-controlled-b128-t16-notimers/profile_report.json`.

Plain read: the scalar-ray source-backed sidecar can be hooked into a
trainer-shaped diagnostic loop now. The first measured local tax is source
snapshot adaptation plus observation production. Replay write/read is still
tiny at these chunk sizes. Default source setup can produce zero body circles,
so `controlled_trail` is the better current optimizer baseline when timing
ray/body geometry.

The optional repo-native PPO learner smoke skipped locally because `torch` is
not importable in this environment and the project does not declare it as a
dependency. That is a local measurement blocker, not evidence that the trainer
shape is wrong.

## Current Bottleneck Read

- The old Pong slice suggested synchronous LightZero can be collect/eval/search
  heavy, not learner-GPU heavy. Treat that as historical shape evidence only.
- For CurvyTron scalar-ray sidecar chunks, source adapter and observation are
  now the big local buckets; replay write is still tiny at these chunk sizes.
- The visual no-train adapter smoke now passes locally and in the installed
  Modal LightZero/DI-engine runtime for
  `curvyzero_debug_visual_tensor_lightzero`. This proves setup plumbing only:
  import/config/env-factory/direct reset/step, real `BaseEnvTimestep`,
  LightZero-facing `float32[1,64,64]` payload, model stack target
  `float32[4,64,64]`, action space `3`, and no ALE identity.
- Local direct adapter timing for `512` debug visual steps measured
  `env_step_total=0.0690s`, about `7416` transitions/s. This is still not a full
  iteration because it excludes trainer frame-stack consumption, policy/search,
  replay/sample, learner, checkpoint, and eval.
- Latest installed CurvyTron debug visual LightZero profile:
  Modal app `ap-8Y4Ezpvfx7B12WHdtyFeei`, `steps=4`, `num_simulations=2`,
  LightZero `0.2.0`, DI-engine `0.5.3`, CPU policy. It used
  `MuZeroPolicy.eval_mode.forward` for search with observation `[1,4,64,64]`,
  built visual replay rows, sampled batch `[4,4,64,64]`, and ran
  `MuZeroPolicy.learn_mode.forward` for forward/loss under a no-op patch:
  optimizer, scheduler, and target updates blocked,
  `model_parameters_changed=false`, `model_state_restored=true`. Timing:
  elapsed `6.524s`, setup `6.416s`, policy/search `0.0381s` for 4 eval/search
  calls, env step/render/stack `0.00128s`, replay row `0.000184s`, sample
  `0.000077s`, learner forward/loss `0.0674s`.
- The next missing measurement sequence is the real LightZero collector/replay
  buffer/learn-mode boundary: `collector.collect`, `MuZeroGameBuffer`
  push/sample/target construction, `policy.learn_mode.forward` or an equivalent
  no-step loss hook, checkpoint/eval only when intentionally included.

## Next LightZero Timing Harness

Do not launch Pong for CurvyTron optimizer work. Do not launch a full CurvyTron
training run just to profile. The first bounded CurvyTron visual profile exists;
the next rung should copy more of the upstream `train_muzero` component setup,
then time:

- setup/import/config/env-manager creation;
- one `collector.collect(...)`;
- replay `push_game_segments(...)` and `remove_oldest_data_to_fit(...)`;
- `replay_buffer.sample(batch_size, policy)`;
- one to five `learner.train(train_data, collector.envstep)` calls;
- optional checkpoint/eval only when explicitly measuring that tax;
- denominator counts: env steps, game segments, replay samples, learner train
  calls, train-iter delta, and GPU samples.

Keep Modal disaggregation out of this rung. If the profile later shows idle
actors, idle learner, or tiny unbatched GPU calls, test overlap or batched
inference inside one container before splitting hot-path pieces across Modal
services.
