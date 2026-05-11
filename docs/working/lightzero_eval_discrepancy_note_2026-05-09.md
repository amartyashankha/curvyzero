# LightZero Eval Discrepancy Note - 2026-05-09

Purpose: explain why the completed `8192` faithful-short Pong eval can show weak
improvement while the active `32768` live eval on `iteration_0` looks bad, and
what to make of the CPU/GPU live-eval difference on that same initial
checkpoint.

Sources used: `docs/working/lightzero_live_eval_notes_2026-05-09.md`,
`docs/experiments/2026-05-09-modal-lightzero-exact-faithful-short-8192-relpath.md`,
`docs/working/lightzero_live_gpu_eval_loop_2026-05-09.md`,
`docs/working/training_state_index_2026-05-09.md`,
`artifacts/local/lightzero-live-eval/manifest_custom_steps512_seed0_20260509T235146Z.json`,
and `artifacts/local/lightzero-live-gpu-eval/manifest_low_steps512_seed0_20260509T235540Z.json`.

## Plain Answer

There is no real contradiction yet.

The `8192` run compares a starting checkpoint to a later completed checkpoint.
The active `32768` live eval only measured `iteration_0`, which is the starting
checkpoint for that attempt. A bad `iteration_0` result is expected baseline
evidence, not evidence that the active run failed to improve.

Say it plainly: `8192 final` versus `32768 iteration_0` is not a regression
comparison. It is later checkpoint versus starting checkpoint from different
attempts.

The `8192` improvement is also weak. Under the corrected stock-ish eval,
`iteration_0` was manual `-13` and stock `-13`, while final `iteration_3697`
was manual `-5` and stock `-8`. It had one positive reward and fewer nonzero
reward events. That is a small before/after signal from one seed and two
checkpoints, not solved Pong.

## What Is Comparable

- Same broad lane: installed `LightZero==0.2.0`, official Atari
  `PongNoFrameskip-v4`, faithful-short wrapper, strict checkpoint load, and no
  model fallback.
- Same score shape for the main reads: 512-step eval cap, `max_episode_steps=512`,
  `num_simulations=50`, `evaluator_env_num=3`, `collector_env_num=8`,
  `batch_size=256`, and `game_segment_length=400`.
- Same validity gates: `strict_load=true`, `fallback_used=false`, and stock
  evaluator available.
- The active `32768` `iteration_0` is comparable to other `iteration_0`
  baselines as an initial policy read.

## What Is Not Comparable

- `8192 iteration_3697` is not comparable to `32768 iteration_0` as a learning
  result. One is after training; the other is before meaningful training.
- A completed attempt is not the same evidence type as an in-flight attempt with
  no later periodic checkpoint evaluated yet.
- Manual return is not the primary score when stock evaluator output exists.
  Manual/stock mismatch remains an eval-harness parity warning.
- CPU and GPU live evals are not a learning comparison. They are placement
  checks on the same untrained checkpoint.

## Survival-First Readout

For weak Pong policies, return alone is too blunt. Read future rows in this
order:

1. Same-run baseline and checkpoint ref.
2. Stock return, then manual/raw return.
3. Steps survived and whether that beats same-run `iteration_0`.
4. Nonzero reward count, positive reward count, and reward timing.
5. Action entropy/collapse and CPU/GPU placement.

This reporting rule does not change the training objective. It makes survival
impossible to bury while keeping sparse score reward as the main objective.

## CPU vs GPU Live Eval On `32768 iteration_0`

Both live evals loaded the same checkpoint strictly and used no fallback.
Both say the initial checkpoint is bad.

| Path | Manual return | Stock return | Action read | Stock/manual prefix |
| --- | ---: | ---: | --- | --- |
| CPU | `-13` | `-12` | mixed actions, dominant action `0` share `0.519531`, entropy `0.809641` | `false` |
| GPU | `-13` | `-13` | collapsed to action `2` for all `512` steps, entropy `0.0` | `true` |

The CPU/GPU difference is real as eval telemetry, but it is not progress
telemetry. At `iteration_0`, the policy is effectively untrained, so small
numerical or evaluator-path differences can flip MCTS tie behavior and produce
very different action histograms while leaving the score equally bad. The
one-point stock-return difference, `-12` versus `-13`, is not meaningful proof
of a better policy.

The GPU result is useful because it proves the cheap GPU eval path can run with
CUDA enabled and strict loading. It also warns that action collapse can appear
immediately on the GPU path, so CPU/GPU parity should not be assumed from one
initial checkpoint.

## What Would Prove Progress

Progress for the active `32768` run would require a later normal checkpoint,
not `ckpt_best`, evaluated against its own `iteration_0` under the same contract:

- strict load true and fallback false;
- stock evaluator return as the main score;
- same 512-step cap and stock-ish knobs;
- fresh recorded pseudo-random eval seed list, not a reused panel;
- better stock return than that run's `iteration_0`;
- positive rewards appearing, fewer negative score events, or both;
- action telemetry that is not just a brittle single-action collapse;
- CPU/GPU agreement on the direction of change, even if exact first actions
  differ.

The minimum useful proof is `32768 iteration_0` versus one later
`iteration_<N>.pth.tar` from the same attempt. The stronger proof is
`iteration_0`, one mid checkpoint, and final over the same freshly sampled,
recorded multi-start eval wave.

## What To Do Next

Do not treat the active `32768 iteration_0` eval as bad news about training. It
is the starting line.

Poll for the next complete `iteration_*.pth.tar` in the active attempt and eval
that checkpoint once with the same stock-ish settings. Keep `ckpt_best` out of
the live quality loop unless its checkpoint state proves it is a real trained
checkpoint.

Read future results in this order: stock return, positive reward count,
negative/nonzero reward count, then action collapse metrics. Keep the manual
episode telemetry beside the stock score for diagnosis, but do not let
manual/stock mismatch become a training conclusion.

For CPU/GPU, rerun parity only when there is a later checkpoint. The useful
question is not whether `iteration_0` picks exactly the same actions on CPU and
GPU; it is whether both placements agree that a later checkpoint improved over
the same-run baseline.

Current checklist:

1. Poll for a later normal `32768` checkpoint.
2. Eval it once under the strict no-fallback 512-step contract.
3. Summarize with baseline deltas versus `32768 iteration_0`.
4. Keep `ckpt_best` out unless state proves it is a trained checkpoint.
5. Keep lane labels explicit: official Atari LightZero control, custom dummy
   Pong bridge/debug, and CurvyTron repo-native target.

No pytest was run. No training code was edited.
