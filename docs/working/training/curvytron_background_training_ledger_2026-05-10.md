# CurvyTron Background Training Ledger - 2026-05-10

Purpose: keep a small cheap batch of native LightZero CurvyTron
`train_muzero` jobs developing artifacts in the background while analysis
continues.

Wrapper used:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Important scope notes:

```text
trainer_entrypoint: lzero.entry.train_muzero
custom two-seat adapter: not used
main path: native LightZero train_muzero CurvyTron visual survival
target reward: survival time
opponent relation: fixed or frozen opponent inside env, not current-policy two-seat self-play
eval rule: reproducible random start panel, parallel survival-steps curve
compute: gpu-l4-t4
checkpoint publishing: live publisher plus final mirror, frequent save_ckpt_after_iter
pytest: not run
launch style: modal run --detach plus wrapper background spawn, not --wait-for-train
```

Safety note: `modal app list` showed an already-active CurvyZero ephemeral app
with many tasks (`ap-36CHuCQOLopsoMQ7hSgrn8`, 48 tasks at the post-launch poll),
so this launch was capped at two jobs instead of three. After correction, the
two live background launches are visible as detached Modal apps with one task
each.

Initial non-detached spawn attempts failed immediately:

```text
fixed initial app: ap-rPoefNhhsWGU94M7M9U6oA
fixed initial function_call_id: fc-01KR9YVDVVP72KWZD09HZQ4P1Y
fixed initial call graph: TERMINATED, RemoteError, no artifact root visible

frozen initial app: ap-VOGQrqa9cpa6lgfyon32sT
frozen initial function_call_id: fc-01KR9YVVE35ZEKG2S62MR7TDV3
frozen initial call graph: TERMINATED, RemoteError, no artifact root visible
```

Those are superseded by the detached attempts below.

## Launch 1 - Fixed-Straight Longer Native Visual Survival

Intent: player-aware debug-visual survival training against env-owned
`fixed_straight`, longer than smoke, different seed from prior s100/s101/s102
runs.

```text
run_id: curvytron-bg-native-fixed-s103-262144
attempt_id: train-gpu-l4t4-bg-fixed-262144x1024-s103-detach-20260510
seed: 103
opponent_policy_kind: fixed_straight
compute: gpu-l4-t4
max_env_step: 262144
max_train_iter: 1024
source_max_steps: 1024
num_simulations: 4
batch_size: 32
save_ckpt_after_iter: 16
Modal launch app id: ap-AEVeMUeogs93Ayq1tvY0cs
Modal launch app status: ephemeral detached, 1 task
spawn status: spawned
function_call_id: fc-01KR9Z21MJFHN2P4BJESV5NYXS
function dashboard: https://modal.com/id/fc-01KR9Z21MJFHN2P4BJESV5NYXS
call graph status at launch poll: PENDING
```

Command:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train --mode train --compute gpu-l4-t4 --seed 103 --run-id curvytron-bg-native-fixed-s103-262144 --attempt-id train-gpu-l4t4-bg-fixed-262144x1024-s103-detach-20260510 --max-env-step 262144 --max-train-iter 1024 --source-max-steps 1024 --collector-env-num 1 --evaluator-env-num 1 --n-evaluator-episode 1 --n-episode 1 --num-simulations 4 --batch-size 32 --save-ckpt-after-iter 16 --opponent-policy-kind fixed_straight
```

Expected artifacts:

```text
summary_ref:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-fixed-s103-262144/attempts/train-gpu-l4t4-bg-fixed-262144x1024-s103-detach-20260510/train/summary.json

action_observability_ref:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-fixed-s103-262144/attempts/train-gpu-l4t4-bg-fixed-262144x1024-s103-detach-20260510/train/action_observability.json

checkpoint root:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-fixed-s103-262144/checkpoints/lightzero

expected checkpoints:
iteration_0.pth.tar, iteration_16.pth.tar, iteration_32.pth.tar, then every
16 learner iterations, plus ckpt_best.pth.tar and a final late checkpoint near
the LightZero stopping boundary.
```

Completed eval:

```text
fixed-straight survival curve, cap 1024, 32 eval starts:
0,128,256,384,512,768,1024,1057

manifest:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-fixed-s103-262144/attempts/train-gpu-l4t4-bg-fixed-262144x1024-s103-detach-20260510/eval/bg_native_fixed_s103_curve_fixedstraight_rand32_20260510/manifest_steps1024_seedsn32_b39991066252_20260510T222534Z.json
```

| iteration | mean steps | median | min | max | capped |
|---:|---:|---:|---:|---:|---:|
| 0 | 159.594 | 142.0 | 100 | 323 | 0 |
| 128 | 186.688 | 182.5 | 40 | 362 | 0 |
| 256 | 163.906 | 147.0 | 51 | 323 | 0 |
| 384 | 170.000 | 150.0 | 40 | 362 | 0 |
| 512 | 153.250 | 139.5 | 42 | 303 | 0 |
| 768 | 191.906 | 184.5 | 51 | 362 | 0 |
| 1024 | 195.406 | 175.0 | 51 | 362 | 0 |
| 1057 | 194.969 | 191.5 | 63 | 362 | 0 |

Read: modest late lift, but no 1024-step caps.

## Launch 2 - Frozen s42 Checkpoint Opponent Native Visual Survival

Intent: native learner-vs-frozen-LightZero-checkpoint lane using a checkpoint
already documented as compatible and smoke-safe: s42 `iteration_293`.

Compatible checkpoint used:

```text
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar
snapshot_ref: curvytron_visual_survival_s42_iteration_293
```

```text
run_id: curvytron-bg-native-frozen-s42iter293-s104-131072
attempt_id: train-gpu-l4t4-bg-frozen-s42iter293-131072x512-s104-detach-20260510
seed: 104
opponent_policy_kind: frozen_lightzero_checkpoint
compute: gpu-l4-t4
max_env_step: 131072
max_train_iter: 512
source_max_steps: 1024
num_simulations: 4
batch_size: 32
save_ckpt_after_iter: 8
Modal launch app id: ap-US1mvUj7Npav78Fh0SAUGm
Modal launch app status: ephemeral detached, 1 task
spawn status: spawned
function_call_id: fc-01KR9Z2WR1614RN4DYT9SQ901W
function dashboard: https://modal.com/id/fc-01KR9Z2WR1614RN4DYT9SQ901W
call graph status at launch poll: PENDING
```

Command:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train --mode train --compute gpu-l4-t4 --seed 104 --run-id curvytron-bg-native-frozen-s42iter293-s104-131072 --attempt-id train-gpu-l4t4-bg-frozen-s42iter293-131072x512-s104-detach-20260510 --max-env-step 131072 --max-train-iter 512 --source-max-steps 1024 --collector-env-num 1 --evaluator-env-num 1 --n-evaluator-episode 1 --n-episode 1 --num-simulations 4 --batch-size 32 --save-ckpt-after-iter 8 --opponent-policy-kind frozen_lightzero_checkpoint --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar --checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar --snapshot-ref curvytron_visual_survival_s42_iteration_293
```

Expected artifacts:

```text
summary_ref:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-frozen-s42iter293-s104-131072/attempts/train-gpu-l4t4-bg-frozen-s42iter293-131072x512-s104-detach-20260510/train/summary.json

action_observability_ref:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-frozen-s42iter293-s104-131072/attempts/train-gpu-l4t4-bg-frozen-s42iter293-131072x512-s104-detach-20260510/train/action_observability.json

checkpoint root:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-frozen-s42iter293-s104-131072/checkpoints/lightzero

expected checkpoints:
iteration_0.pth.tar, iteration_8.pth.tar, iteration_16.pth.tar, then every
8 learner iterations, plus ckpt_best.pth.tar and a final late checkpoint near
the LightZero stopping boundary.
```

Completed eval:

```text
matched frozen-s42-iteration293 survival curve, cap 1024, 32 eval starts:
0,64,128,192,256,320,384,448,512,518

opponent_checkpoint_ref:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar

opponent_snapshot_ref: curvytron_visual_survival_s42_iteration_293

manifest:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-frozen-s42iter293-s104-131072/attempts/train-gpu-l4t4-bg-frozen-s42iter293-131072x512-s104-detach-20260510/eval/bg_native_frozen_s104_curve_matched_s42iter293_rand32_20260510/manifest_steps1024_seedsn32_b39991066252_20260510T222924Z.json
```

| iteration | mean steps | median | min | max | capped |
|---:|---:|---:|---:|---:|---:|
| 0 | 162.625 | 139.5 | 45 | 326 | 0 |
| 64 | 255.344 | 196.5 | 48 | 1024 | 2 |
| 128 | 173.094 | 141.0 | 45 | 382 | 0 |
| 192 | 271.750 | 192.5 | 47 | 1024 | 1 |
| 256 | 214.156 | 189.0 | 45 | 588 | 0 |
| 320 | 206.875 | 155.5 | 44 | 753 | 0 |
| 384 | 185.594 | 149.5 | 45 | 475 | 0 |
| 448 | 176.531 | 144.5 | 45 | 547 | 0 |
| 512 | 269.688 | 184.0 | 48 | 1024 | 2 |
| 518 | 182.906 | 146.0 | 45 | 448 | 0 |

Read: matched-opponent spikes appear, including five capped episodes across
three checkpoints, but the final checkpoint falls back.

## 2026-05-11 Long Native Eval Results

Main metric: steps survived. Score is secondary here. Treat 1024-step caps as
useful supporting evidence, not the only signal.

All four completed evals used 32 random starts and a 1024-step cap.

### s203 Fixed Straight

```text
run_id: curvytron-bg-native-fixed-s203-1048576
attempt_id: train-gpu-l4t4-bg-fixed-1048576x4096-s203-save16-20260510
```

| checkpoint | mean steps | median | capped |
|---|---:|---:|---:|
| `iteration_0` | 202.281 | 191.5 | 0 |
| `iteration_512` | 156.750 | 137.5 | 0 |
| `iteration_1024` | 164.844 | 137.5 | 0 |
| `iteration_1536` | 170.938 | 163.5 | 0 |
| `iteration_2048` | 170.594 | 157.5 | 0 |
| `iteration_2560` | 161.750 | 150.5 | 0 |
| `iteration_3072` | 169.906 | 157.5 | 0 |
| `iteration_3584` | 173.125 | 158.5 | 0 |
| `iteration_4096` | 164.875 | 150.5 | 0 |
| `iteration_4106` | 186.500 | 182.5 | 0 |

Read: no durable improvement.

### s205 Fixed Straight

```text
run_id: curvytron-bg-native-fixed-s205-1048576
attempt_id: train-gpu-l4t4-bg-fixed-1048576x4096-s205-detach-20260510
```

| checkpoint | mean steps | median | capped |
|---|---:|---:|---:|
| `iteration_0` | 159.594 | 142.0 | 0 |
| `iteration_512` | 189.469 | 178.0 | 0 |
| `iteration_1024` | 159.719 | 147.0 | 0 |
| `iteration_1536` | 167.875 | 147.0 | 0 |
| `iteration_2048` | 180.469 | 160.5 | 0 |
| `iteration_2560` | 181.688 | 176.5 | 0 |
| `iteration_3072` | 168.188 | 147.0 | 0 |
| `iteration_3584` | 166.000 | 156.0 | 0 |
| `iteration_4096` | 194.688 | 183.5 | 0 |
| `iteration_4125` | 190.531 | 178.0 | 0 |

Read: modest lift, still weak.

### s204 Frozen s47 iteration_200

```text
run_id: curvytron-bg-native-frozen-s47iter200-s204-1048576
attempt_id: train-gpu-l4t4-bg-frozen-s47iter200-1048576x4096-s204-save8-20260510
```

| checkpoint | mean steps | median | capped |
|---|---:|---:|---:|
| `iteration_0` | 168.250 | 133.5 | 0 |
| `iteration_512` | 460.000 | 240.0 | 9 |
| `iteration_1024` | 312.094 | 162.5 | 3 |
| `iteration_1536` | 289.062 | 114.5 | 4 |
| `iteration_2048` | 356.438 | 190.0 | 5 |
| `iteration_2560` | 462.906 | 271.5 | 7 |
| `iteration_3072` | 498.969 | 454.5 | 9 |
| `iteration_3584` | 334.219 | 183.0 | 4 |
| `iteration_4096` | 329.500 | 188.0 | 3 |
| `iteration_4105` | 318.844 | 183.0 | 4 |

Read: real matched-opponent survival signal, but unstable. It needs a broader
opponent check.

### s206 Frozen s92 iteration_434

```text
run_id: curvytron-bg-native-frozen-s92iter434-s206-524288
attempt_id: train-gpu-l4t4-bg-frozen-s92iter434-524288x2048-s206-detach-20260510
```

| checkpoint | mean steps | median | capped |
|---|---:|---:|---:|
| `iteration_0` | 510.125 | 354.5 | 12 |
| `iteration_256` | 286.938 | 192.5 | 3 |
| `iteration_512` | 252.406 | 192.5 | 2 |
| `iteration_768` | 186.812 | 141.0 | 0 |
| `iteration_1024` | 206.156 | 170.5 | 0 |
| `iteration_1280` | 201.844 | 131.0 | 2 |
| `iteration_1536` | 251.938 | 132.0 | 1 |
| `iteration_1792` | 227.125 | 143.0 | 2 |
| `iteration_2048` | 284.781 | 147.5 | 3 |
| `iteration_2099` | 189.812 | 138.0 | 0 |

Read: got worse from a strong initial checkpoint.

Next decision: fixed-opponent runs are not enough. Matched frozen-opponent runs
can show signal, but they may overfit to one opponent. For best checkpoints,
the next eval should use a small panel: `fixed_straight`, frozen s47
`iteration_200`, and frozen s92 `iteration_434`.

## Skipped

Skipped a third launch because an existing CurvyZero ephemeral app already had
many active tasks and the request explicitly asked for a small cheap batch, not
an absurd fanout.

Frozen checkpoint setup was not skipped. The s42 `iteration_293` checkpoint was
available in `curvyzero-runs` and had prior same-day smoke evidence in
`curvytron_native_frozen_opponent_probe_2026-05-10.md` and
`curvytron_snapshot_opponent_interface_2026-05-10.md`.

## Live Status Check

Checked after launch:

```text
curvytron-bg-native-fixed-s103-262144:
  status_heartbeat: running
  stage: before_train_muzero
  heartbeat_at: 2026-05-10T22:10:33.154330Z
  checkpoint root currently contains: ckpt_best.pth.tar

curvytron-bg-native-frozen-s42iter293-s104-131072:
  status_heartbeat: running
  stage: before_train_muzero
  heartbeat_at: 2026-05-10T22:11:01.331252Z
  checkpoint root currently contains: ckpt_best.pth.tar
```

These are the two real background jobs to watch.

## Superseded Main-Thread Hail Mary Attempts

Two larger main-thread launches were attempted without `--detach`:

```text
fixed:
  run_id: curvytron-visual-survival-player-aware-fixed-s201-hailmary1048576
  attempt_id: train-gpu-l4t4-player-aware-fixed-1048576x4096-s201-hailmary-20260510
  app: ap-3K39J4AdVEfdWJj9nDM8jo
  function_call_id: fc-01KR9Z13XXCMWQTGMAP4RGAECB

frozen:
  run_id: curvytron-visual-survival-frozen-s47iter200-s202-hailmary1048576
  attempt_id: train-gpu-l4t4-frozen-s47iter200-1048576x4096-s202-hailmary-20260510
  app: ap-mX4GM4Gq7ZOAWeaS7jk8xl
  function_call_id: fc-01KR9Z1GSF3SS1CF6YK28MHG03
```

Follow-up volume checks found no attempt `train/` artifact roots for these two
run ids. Do not count them as live background runs unless a later Modal
function-call check proves otherwise. The tracked background jobs are the
detached `s103` and `s104` launches above.

## Additional Native Long Launches - 2026-05-10 22:23 UTC

No pytest was run.

Checkpoint cadence rule for this wave:

```text
fixed-straight long run: explicit --save-ckpt-after-iter 16
frozen-opponent long run: explicit --save-ckpt-after-iter 8
launched without explicit cadence: none
kill/relaunch needed for missing cadence: no
```

Wrapper support confirmed before launch:

```text
native entrypoint: lzero.entry.train_muzero
launch path: modal run --detach plus local entrypoint train_fn.spawn(...)
compute: gpu-l4-t4, backed by cheap GPU resource ["L4", "T4"]
GPU timeout: 8 hours
```

The frozen-opponent launch uses s92 `iteration_434` because
`curvytron_next_native_experiment_decision_2026-05-10.md` confirmed it as the
strongest current matched frozen-opponent checkpoint:

```text
s92 matched eval, 32 seeds:
iteration_0 mean steps: 151.781
iteration_384 mean steps: 417.031
iteration_434 mean steps: 500.438

checkpoint verified present:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-refresh-s47iter200-s92-65536/checkpoints/lightzero/iteration_434.pth.tar
```

### Launch 3 - Fixed-Straight Long Native Visual Survival

```text
run_id: curvytron-bg-native-fixed-s205-1048576
attempt_id: train-gpu-l4t4-bg-fixed-1048576x4096-s205-detach-20260510
seed: 205
opponent_policy_kind: fixed_straight
compute: gpu-l4-t4
max_env_step: 1048576
max_train_iter: 4096
source_max_steps: 1024
num_simulations: 4
batch_size: 32
save_ckpt_after_iter: 16
Modal launch app id: ap-5d6TinHWOsxGb93fRNYr3H
Modal launch app status at poll: ephemeral detached, 1 task
spawn status: spawned
function_call_id: fc-01KR9ZS2DF5YYSAQME72W815RV
function dashboard: https://modal.com/id/fc-01KR9ZS2DF5YYSAQME72W815RV
heartbeat modal_task_id: ta-01KR9ZS3DPZEXWBAK034009RDD
heartbeat status: running
heartbeat stage: before_train_muzero
heartbeat_at: 2026-05-10T22:23:07.964291Z
checkpoint root currently contains: ckpt_best.pth.tar
```

Command:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train --mode train --compute gpu-l4-t4 --seed 205 --run-id curvytron-bg-native-fixed-s205-1048576 --attempt-id train-gpu-l4t4-bg-fixed-1048576x4096-s205-detach-20260510 --max-env-step 1048576 --max-train-iter 4096 --source-max-steps 1024 --collector-env-num 1 --evaluator-env-num 1 --n-evaluator-episode 1 --n-episode 1 --num-simulations 4 --batch-size 32 --save-ckpt-after-iter 16 --opponent-policy-kind fixed_straight
```

Expected artifacts:

```text
summary_ref:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-fixed-s205-1048576/attempts/train-gpu-l4t4-bg-fixed-1048576x4096-s205-detach-20260510/train/summary.json

action_observability_ref:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-fixed-s205-1048576/attempts/train-gpu-l4t4-bg-fixed-1048576x4096-s205-detach-20260510/train/action_observability.json

status_heartbeat_ref:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-fixed-s205-1048576/attempts/train-gpu-l4t4-bg-fixed-1048576x4096-s205-detach-20260510/train/status_heartbeat.json

checkpoint root:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-fixed-s205-1048576/checkpoints/lightzero

expected checkpoints:
iteration_0.pth.tar, iteration_16.pth.tar, iteration_32.pth.tar, then every
16 learner iterations, plus ckpt_best.pth.tar and a final late checkpoint near
the LightZero stopping boundary.
```

Eval later:

```text
fixed-straight survival curve, cap 1024, 32 eval starts if budget allows:
0,128,256,384,512,640,768,896,1024,1536,2048,2560,3072,3584,4096/final

also inspect action_observability for action collapse and source-step survival
distribution across the long run.
```

### Launch 4 - Frozen s92 iteration_434 Opponent Long Native Visual Survival

```text
run_id: curvytron-bg-native-frozen-s92iter434-s206-524288
attempt_id: train-gpu-l4t4-bg-frozen-s92iter434-524288x2048-s206-detach-20260510
seed: 206
opponent_policy_kind: frozen_lightzero_checkpoint
opponent_checkpoint_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-refresh-s47iter200-s92-65536/checkpoints/lightzero/iteration_434.pth.tar
snapshot_ref/opponent_snapshot_ref: curvytron_visual_survival_s92_iteration_434
compute: gpu-l4-t4
max_env_step: 524288
max_train_iter: 2048
source_max_steps: 1024
num_simulations: 4
batch_size: 32
save_ckpt_after_iter: 8
Modal launch app id: ap-E5ugDmidqrq6lWX0r3qLqz
Modal launch app status at poll: ephemeral detached, 1 task
spawn status: spawned
function_call_id: fc-01KR9ZSRDZB13FY9MMHKDR2CXD
function dashboard: https://modal.com/id/fc-01KR9ZSRDZB13FY9MMHKDR2CXD
heartbeat modal_task_id: ta-01KR9ZSS4VWENVHMHRD1JMDS5C
heartbeat status: running
heartbeat stage: before_train_muzero
heartbeat_at: 2026-05-10T22:23:31.148553Z
checkpoint root currently contains: ckpt_best.pth.tar
```

Command:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train --mode train --compute gpu-l4-t4 --seed 206 --run-id curvytron-bg-native-frozen-s92iter434-s206-524288 --attempt-id train-gpu-l4t4-bg-frozen-s92iter434-524288x2048-s206-detach-20260510 --max-env-step 524288 --max-train-iter 2048 --source-max-steps 1024 --collector-env-num 1 --evaluator-env-num 1 --n-evaluator-episode 1 --n-episode 1 --num-simulations 4 --batch-size 32 --save-ckpt-after-iter 8 --opponent-policy-kind frozen_lightzero_checkpoint --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-refresh-s47iter200-s92-65536/checkpoints/lightzero/iteration_434.pth.tar --checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-refresh-s47iter200-s92-65536/checkpoints/lightzero/iteration_434.pth.tar --snapshot-ref curvytron_visual_survival_s92_iteration_434
```

Expected artifacts:

```text
summary_ref:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-frozen-s92iter434-s206-524288/attempts/train-gpu-l4t4-bg-frozen-s92iter434-524288x2048-s206-detach-20260510/train/summary.json

action_observability_ref:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-frozen-s92iter434-s206-524288/attempts/train-gpu-l4t4-bg-frozen-s92iter434-524288x2048-s206-detach-20260510/train/action_observability.json

status_heartbeat_ref:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-frozen-s92iter434-s206-524288/attempts/train-gpu-l4t4-bg-frozen-s92iter434-524288x2048-s206-detach-20260510/train/status_heartbeat.json

checkpoint root:
training/lightzero-curvytron-visual-survival/curvytron-bg-native-frozen-s92iter434-s206-524288/checkpoints/lightzero

expected checkpoints:
iteration_0.pth.tar, iteration_8.pth.tar, iteration_16.pth.tar, then every
8 learner iterations, plus ckpt_best.pth.tar and a final late checkpoint near
the LightZero stopping boundary.
```

Eval later:

```text
fixed-straight baseline survival curve, cap 1024:
0,64,128,192,256,320,384,448,512,768,1024,1536,2048/final

matched frozen-s92-iteration434 survival curve, cap 1024, same eval points.

broader opponent panel after matched eval:
fixed_straight, s47 iteration_200, s92 iteration_434.

inspect action_observability for mostly-one-turn high survival rows; the s92
signal may still be matched-opponent exploitation rather than general survival.
```

## 2026-05-10 Long Run Checkpoint Verification

Modal volume check:

```text
uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero
```

| run | attempt | save cadence | latest checkpoint | completion | completed long eval points |
|---|---|---:|---:|---|---|
| `curvytron-bg-native-fixed-s203-1048576` | `train-gpu-l4t4-bg-fixed-1048576x4096-s203-save16-20260510` | 16 | `iteration_4106` | complete; `summary.json` present | `0,512,1024,1536,2048,2560,3072,3584,4096,4106` |
| `curvytron-bg-native-frozen-s47iter200-s204-1048576` | `train-gpu-l4t4-bg-frozen-s47iter200-1048576x4096-s204-save8-20260510` | 8 | `iteration_4105` | complete; `summary.json` present | `0,512,1024,1536,2048,2560,3072,3584,4096,4105` |
| `curvytron-bg-native-fixed-s205-1048576` | `train-gpu-l4t4-bg-fixed-1048576x4096-s205-detach-20260510` | 16 | `iteration_4125` | complete; `summary.json` present | `0,512,1024,1536,2048,2560,3072,3584,4096,4125` |
| `curvytron-bg-native-frozen-s92iter434-s206-524288` | `train-gpu-l4t4-bg-frozen-s92iter434-524288x2048-s206-detach-20260510` | 8 | `iteration_2099` | complete; `summary.json` present | `0,256,512,768,1024,1280,1536,1792,2048,2099` |

Compact checkpoint ranges:

```text
s203 fixed: 258 iteration checkpoints, 0..4106, cadence 16 plus final
s204 frozen s47iter200: 515 iteration checkpoints, 0..4105, cadence 8 plus final
s205 fixed: 259 iteration checkpoints, 0..4125, cadence 16 plus final
s206 frozen s92iter434: 264 iteration checkpoints, 0..2099, cadence 8 plus final
```

## 2026-05-11 Self-Play Reset

Reality correction:

```text
native train_muzero fixed/frozen runs: not self-play
actual current-policy self-play path: curvytron_two_seat_lightzero_train_smoke
self-play meaning here: one live LightZero policy chooses both players' actions before the env step
reward: survival reward by default, 1 while alive and 0 when dead
episode cap for new long runs: max_ticks 16384
start state: generated reset seeds from the env RNG, not one repeated fixed start
old scattershot apps: stopped
old local Pong trainer processes: none found with pgrep at cleanup time
```

Code changes made before the long wave:

```text
src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py
src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py
```

Plain change list:

- The two-seat path now uses generated reset seeds for the initial reset and
  autoresets, so rows do not replay the same start.
- Modal two-seat runs now accept and pass `max_ticks`; long runs use `16384`.
- Long-run output is bounded so the job does not die from writing huge replay
  tensors.
- Summaries now include completed episode duration numbers from the trainer.
- Two simple knobs exist for the next wave: `dead_reward` and
  `action_noop_probability`.

Sanity checks passed:

| run_id | read |
|---|---|
| `curvytron-two-seat-selfplay-sanity-patched-20260510` | completed; checkpoints `iteration_0..2`; optimizer changed model weights; duration summary present |
| `curvytron-two-seat-selfplay-knob-sanity-20260510` | completed; checkpoints `iteration_0..2`; dead reward and no-op action-noise knobs worked; optimizer changed model weights |

Stopped apps from the earlier bad/obsolete wave:

```text
ap-lyFtI9Au8yGjxL5BkGPqEA
ap-PtXjIM98ArHiHJTLvY89KR
ap-ogsp7cRmdkGD3wmPdYdgrg
ap-Jmpz4PETLxCismoVPrwgG5
ap-E0E6wFDe9yVOv4m1QMf5An
```

Active long self-play wave:

| run_id | app | knobs | purpose |
|---|---|---|---|
| `curvytron-two-seat-selfplay-reference-v2-b32x128x128-u4-sim4-20260510` | `ap-lEOXcSQhKxtVam0TfGrUCD` | alive `1.0`, dead `0.0`, noise `0.0` | clean reference |
| `curvytron-two-seat-selfplay-deadpenalty005-b32x128x128-u4-sim4-20260510` | `ap-IkQqiCokpAdsVRuxb3FDmz` | alive `1.0`, dead `-0.05`, noise `0.0` | small death penalty |
| `curvytron-two-seat-selfplay-noop003-b32x128x128-u4-sim4-20260510` | `ap-w9eq7KCB4AMLnXJI60hQQA` | alive `1.0`, dead `0.0`, noise `0.03` | small action robustness pressure |
| `curvytron-two-seat-selfplay-strongupdate-b32x128x128-u8-sim4-20260510` | `ap-O2S9dTI3daDOrnYiHIbkRy` | alive `1.0`, dead `0.0`, noise `0.0`, 8 updates | stronger learner pressure |

All four active runs got past Modal setup/import. Next check: wait for
completion or inspect checkpoints/summaries in the volume. Do not launch the
older-checkpoint opponent idea yet; that is a larger off-policy change and
needs a cleaner design.

2026-05-11 02:05 EDT update:

- The first four long self-play runs are alive on Modal and each has already
  saved `iteration_0` and `iteration_1`. This means the patched path is not just
  booting; it is collecting, updating, and writing checkpoints.
- Four more creative variants were launched after the first checkpoint check.
  These are not just different run numbers; they change reward scale, small
  action no-op noise, search work, or batch size.
- Keep judging runs by completed game length first. Score can matter later, but
  for CurvyTron the clean first signal is: are games lasting longer over
  training?

Additional active long self-play variants:

| run_id | app | knobs | purpose |
|---|---|---|---|
| `curvytron-two-seat-selfplay-scaledreward001-dead1-b32x128x128-u4-sim4-20260510` | `ap-JilDn3tbEU3tx8C8cD1cPy` | alive `0.01`, dead `-1.0`, noise `0.0` | smaller value scale with real death penalty |
| `curvytron-two-seat-selfplay-scaledreward001-noop001-b32x128x128-u4-sim4-20260510` | `ap-eeWphsmybK8AjX8U1hlCFo` | alive `0.01`, dead `-1.0`, noise `0.01` | same reward plus tiny no-op robustness pressure |
| `curvytron-two-seat-selfplay-scaledreward001-sim8-b32x96x128-u4-20260510` | `ap-o3KCjT8Pg2fL0Padmgwp40` | alive `0.01`, dead `-1.0`, search simulations `8` | check whether more search helps collection |
| `curvytron-two-seat-selfplay-scaledreward001-b64x96x128-u4-sim4-20260510` | `ap-ruyCzw4V2EteMa9yvcA1XY` | alive `0.01`, dead `-1.0`, batch `64` | check whether wider collection helps signal |

2026-05-11 02:07 EDT checkpoint check:

| run_id | checkpoint state |
|---|---|
| `curvytron-two-seat-selfplay-scaledreward001-dead1-b32x128x128-u4-sim4-20260510` | `iteration_0`, `iteration_1`, latest, best |
| `curvytron-two-seat-selfplay-scaledreward001-noop001-b32x128x128-u4-sim4-20260510` | `iteration_0`, `iteration_1`, latest, best |
| `curvytron-two-seat-selfplay-scaledreward001-sim8-b32x96x128-u4-20260510` | `iteration_0`, `iteration_1`, latest, best |
| `curvytron-two-seat-selfplay-scaledreward001-b64x96x128-u4-sim4-20260510` | `iteration_0`, latest, best; still running, likely slower because batch is larger |

Current interpretation: the long self-play wave is not crashing at startup and
is saving checkpoints. The next useful data is not another launch; it is a
later checkpoint curve showing whether completed game length rises.

Rules to keep from here:

- Do not treat run-number changes as meaningful experiments.
- Do not over-focus on one repeated start. Training resets should vary starts;
  evaluation can use a reproducible generated set of starts.
- Do not call fixed-opponent CurvyTron training “self-play”.
- Do not launch older-checkpoint opponents until the current-policy path is
  checked. If one player uses an old policy, only update from the live policy's
  own rows unless the training design explicitly handles off-policy data.

## 2026-05-11 Long Hail Mary Self-Play Wave

Correction: one accidental long app was launched with the old “save every loop”
behavior and was stopped immediately:

```text
ap-Ug42KiB47pciT7qszlNQ74
```

The trainer now has:

```text
checkpoint_every_iterations
action_noop_warmup_iterations
```

Long-run rule: do not save every loop. Save often enough to inspect progress,
but not so often that checkpoint I/O becomes the training job.

Active long Hail Mary runs:

| run_id | app | checkpoint cadence | setup |
|---|---:|---:|---|
| `curvytron-two-seat-selfplay-hailmary-basic-survival001-b32-long-u8-sim4-20260510` | `ap-Eb5LgwqSVsFlW0807Nqj06` | every 16 loops | alive `0.01`, dead `0.0`, no action noise |
| `curvytron-two-seat-selfplay-hailmary-deadpenalty-survival001-b32-long-u8-sim4-20260510` | `ap-9700rCXQ1W8SwullPS09Gs` | every 16 loops | alive `0.01`, dead `-1.0`, no action noise |
| `curvytron-two-seat-selfplay-hailmary-noopwarm002-survival001-b32-long-u8-sim4-20260510` | `ap-htxYyAi5YuoJPcmu9AZU87` | every 16 loops | alive `0.01`, dead `-1.0`, no-op action chance ramps from `0.0` to `0.02` over 512 loops |

Common settings:

```text
batch_size: 32
outer_iterations: 16384
collect_steps_per_iteration: 256
updates_per_iteration: 8
num_simulations: 4
learner_sample_size: 1024
max_replay_rows: 32768
max_ticks: 65536
```

Startup check:

```text
all three active Hail Mary runs saved iteration_0
next useful check: wait for iteration_16
main signal to inspect later: completed game length over checkpoints
```

Shorter self-play variants are still useful as speed probes. At 2026-05-11
02:20 EDT, example checkpoint counts were:

```text
reference v2: iteration_0..6
deadpenalty005: iteration_0..7
scaledreward001-dead1: iteration_0..5
scaledreward001 batch64: iteration_0..2
```

Rough expectation: the long runs may need tens of minutes to reach
`iteration_16`, especially because they collect more rows and do more learner
updates per loop than the shorter probes.

## 2026-05-11 Live-Logged Replacement Runs

Problem fixed: a never-ending training run cannot rely on end-of-run
`summary.json` for visibility. Checkpoints are also too expensive to use as the
only progress signal.

Code change:

```text
src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py
src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py
```

New behavior:

- Every loop can emit one small `TRAIN_PROGRESS` JSON line.
- The same progress data is appended to
  `training/lightzero-curvytron-visual-survival/<run>/attempts/<attempt>/train/progress.jsonl`.
- Modal commits the progress file each loop for live volume inspection.
- Checkpoints stay sparse: every 64 loops for open-ended runs, every 16 loops
  for the regular wave.
- Starts still vary through generated reset seeds.

Stopped training apps from the pre-progress wave:

```text
ap-O2S9dTI3daDOrnYiHIbkRy
ap-IkQqiCokpAdsVRuxb3FDmz
ap-lEOXcSQhKxtVam0TfGrUCD
ap-w9eq7KCB4AMLnXJI60hQQA
ap-eeWphsmybK8AjX8U1hlCFo
ap-JilDn3tbEU3tx8C8cD1cPy
ap-o3KCjT8Pg2fL0Padmgwp40
ap-ruyCzw4V2EteMa9yvcA1XY
ap-Eb5LgwqSVsFlW0807Nqj06
ap-htxYyAi5YuoJPcmu9AZU87
ap-9700rCXQ1W8SwullPS09Gs
```

The eval app `ap-6y5iNhDKguczyLr6b7douQ` was not stopped.

Active live-logged open-ended runs:

| run_id | app | checkpoint cadence | setup |
|---|---:|---:|---|
| `curvytron-two-seat-selfplay-live-basic-survival001-b32-open-u8-sim4-20260510` | `ap-mz6kMF2x1JIz5HqnNz4bwE` | every 64 loops | alive `0.01`, dead `0.0`, no action noise |
| `curvytron-two-seat-selfplay-live-deadpenalty-survival001-b32-open-u8-sim4-20260510` | `ap-Q0LrVe4oNzkS1omdJ20clC` | every 64 loops | alive `0.01`, dead `-1.0`, no action noise |
| `curvytron-two-seat-selfplay-live-noopwarm002-survival001-b32-open-u8-sim4-20260510` | `ap-FNNdg5PAd9sAQW2gP6Gtt2` | every 64 loops | alive `0.01`, dead `-1.0`, no-op action chance ramps from `0.0` to `0.02` over 512 loops |

Common open-ended settings:

```text
outer_iterations: 1000000
batch_size: 32
collect_steps_per_iteration: 256
updates_per_iteration: 8
num_simulations: 4
learner_sample_size: 1024
max_replay_rows: 32768
max_ticks: 65536
progress_every_iterations: 1
```

Active live-logged regular runs:

| run_id | app | checkpoint cadence | setup |
|---|---:|---:|---|
| `curvytron-two-seat-selfplay-live-regular-basic-survival001-b32-256x128-u4-sim4-20260510` | `ap-Gy6ZOKSBUIwTPHYRpdLHkM` | every 16 loops | alive `0.01`, dead `0.0`, no action noise |
| `curvytron-two-seat-selfplay-live-regular-deadpenalty-survival001-b32-256x128-u4-sim4-20260510` | `ap-rP14HaipEX7T1727lqfGPu` | every 16 loops | alive `0.01`, dead `-1.0`, no action noise |
| `curvytron-two-seat-selfplay-live-regular-noopwarm002-survival001-b32-256x128-u4-sim4-20260510` | `ap-ID72vheH4uWDad6fyMi4ba` | every 16 loops | alive `0.01`, dead `-1.0`, no-op action chance ramps from `0.0` to `0.02` over 64 loops |

Common regular settings:

```text
outer_iterations: 256
batch_size: 32
collect_steps_per_iteration: 128
updates_per_iteration: 4
num_simulations: 4
learner_sample_size: 512
max_replay_rows: 16384
max_ticks: 16384
progress_every_iterations: 1
```

Startup check:

```text
all six replacement runs printed TRAIN_PROGRESS start lines
next check: wait for iteration progress lines, then inspect mean/max completed game length
```

Direct simple baselines:

```text
script: src/curvyzero/training/curvytron_baseline_eval.py
local check: 64 episodes, batch 64, cap 2048
straight mean steps: about 8.0
split_turn mean steps: about 9.1
weave mean steps: about 9.1
random_legal mean steps: about 6.0
mostly_straight mean steps: about 7.9
```

Interpretation: current dumb-policy floor is tiny under the public env settings.
The first training signal to look for is whether learned self-play pushes
completed game length clearly above this floor and continues rising.

## 2026-05-11 Live6 Long Self-Play Batch

Current priority: run a broad long CurvyTron self-play batch and watch survival
length live. This batch supersedes `live`, `live2`, `live3`, `live4`, and
`live5` variants.

Code changes before launch:

```text
src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py
src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py
```

Important behavior:

- Each run uses the same live LightZero policy object for both CurvyTron players.
- Reset starts vary through generated reset seeds.
- The run seed is passed into LightZero config compilation before `MuZeroPolicy`
  is constructed. First learner hashes should still be checked once iteration
  rows land to prove the neural network weights differ across runs.
- Progress is written to Modal volume every loop:
  `progress.jsonl` and `progress_latest.json`.
- Progress is also printed as compact `TRAIN_PROGRESS` lines.
- Checkpoints are sparse: `iteration_0`, then every 6 outer iterations, plus
  `latest.pth.tar` and `ckpt_best.pth.tar`.
- Modal timeout is 12 hours.
- Image noise, when enabled, is only added to normalized visual tensors given to
  the policy/replay. It is clipped to `[0, 1]` and does not change game physics,
  rewards, or action masks.

Active `live6` runs:

| run_id | app | seed | alive | dead | action no-op | warmup loops | image noise std |
|---|---:|---:|---:|---:|---:|---:|---:|
| `curvytron-two-seat-selfplay-live6-clean-survival001-b32-open-u8-sim4-20260510` | `ap-dD6YQkbMZcsJyARxrNRjmZ` | 950 | 0.01 | 0.0 | 0.00 | 0 | 0.00 |
| `curvytron-two-seat-selfplay-live6-deadpenalty-survival001-b32-open-u8-sim4-20260510` | `ap-2on788VPfPYxUbJjS8dNjf` | 951 | 0.01 | -1.0 | 0.00 | 0 | 0.00 |
| `curvytron-two-seat-selfplay-live6-deadpenalty-survival002-b32-open-u8-sim4-20260510` | `ap-v2HIavZoQmQb9M6uz9EHuK` | 952 | 0.02 | -1.0 | 0.00 | 0 | 0.00 |
| `curvytron-two-seat-selfplay-live6-deadpenalty-survival005-b32-open-u8-sim4-20260510` | `ap-jEu1FLUcbbDiu9rf31cuqc` | 953 | 0.05 | -1.0 | 0.00 | 0 | 0.00 |
| `curvytron-two-seat-selfplay-live6-noopwarm005-survival001-b32-open-u8-sim4-20260510` | `ap-ZC74G44VZryGacbSjrZYAg` | 954 | 0.01 | -1.0 | 0.05 | 144 | 0.00 |
| `curvytron-two-seat-selfplay-live6-noopwarm010-survival001-b32-open-u8-sim4-20260510` | `ap-ol5chbXOTqPZh923aLg9Gk` | 955 | 0.01 | -1.0 | 0.10 | 144 | 0.00 |
| `curvytron-two-seat-selfplay-live6-noopwarm015-survival001-b32-open-u8-sim4-20260510` | `ap-6v86fwMUdWqeLW78VMchts` | 956 | 0.01 | -1.0 | 0.15 | 144 | 0.00 |
| `curvytron-two-seat-selfplay-live6-imgnoise001-survival001-b32-open-u8-sim4-20260510` | `ap-XCjeCnhL5565huSi540RwY` | 957 | 0.01 | -1.0 | 0.00 | 0 | 0.01 |
| `curvytron-two-seat-selfplay-live6-imgnoise003-survival001-b32-open-u8-sim4-20260510` | `ap-dae1lGfJMZktVxWnDKfeo0` | 958 | 0.01 | -1.0 | 0.00 | 0 | 0.03 |
| `curvytron-two-seat-selfplay-live6-imgnoise010-survival001-b32-open-u8-sim4-20260510` | `ap-wkLwsOgVRyo4ckb70PgLxH` | 959 | 0.01 | -1.0 | 0.00 | 0 | 0.10 |
| `curvytron-two-seat-selfplay-live6-mixed-img003-noop010-survival001-b32-open-u8-sim4-20260510` | `ap-sk4VmeFyZPOLxXxjrAkyEL` | 960 | 0.01 | -1.0 | 0.10 | 144 | 0.03 |
| `curvytron-two-seat-selfplay-live6-mixed-img010-noop005-survival002-b32-open-u8-sim4-20260510` | `ap-SJAiSkJXLv5Tkqdums8W6O` | 961 | 0.02 | -1.0 | 0.05 | 144 | 0.10 |

Common settings:

```text
batch_size: 32
outer_iterations: 1000000
collect_steps_per_iteration: 256
updates_per_iteration: 8
num_simulations: 4
learner_sample_size: 1024
max_replay_rows: 32768
max_ticks: 65536
checkpoint_every_iterations: 6
progress_every_iterations: 1
progress_commit_every_iterations: 1
compute: Modal CPU-only function, 64 CPU, 64 GiB memory
```

App cleanup:

```text
stopped superseded live5 apps:
ap-Mee7xtbw2ktgpB65NTEOFq
ap-zUFPyXvHkXBW5vIkY4fNo0
ap-0Xkyu7WwVlCNhcbVOwcYZ1
ap-d0JUoqE11n07unvMGLpqsy
ap-pbRM8ig7BjkHo4fVbrUpHr
ap-huJY6YOKF0buAJ4Wg09Qan
```

Do not stop unrelated deployed apps from other projects.

Immediate checks:

```text
live6 clean progress_latest exists and currently has the start row.
next required check: wait for first iteration rows, then compare:
- action_counts are not collapsed to one action
- model_hash_before differs across runs
- model_parameters_changed is true
- mean_completed_episode_steps and max_completed_episode_steps are visible
```

First live6 iteration sanity:

```text
clean run iteration 1:
  action_counts: 0=5503, 1=5462, 2=5419
  per-player counts: balanced
  model_hash_before: 183bae94e0dfd9c2
  model_hash_after: 64590150edbf8966
  model_parameters_changed: true
  mean_completed_episode_steps: 11.31
  max_completed_episode_steps: 42

noopwarm010 run iteration 1:
  action_counts: 0=5503, 1=5565, 2=5316
  effective_action_noop_probability: 0.000694
  model_hash_before: b777274f7fdb06b6
  model_hash_after: e5623db03a3da64c
  model_parameters_changed: true
  mean_completed_episode_steps: 11.20
  max_completed_episode_steps: 47

mixed img003 + noop010 run iteration 1:
  action_counts: 0=5353, 1=5570, 2=5461
  effective_action_noop_probability: 0.000694
  model_hash_before: 849be202f23ed80c
  model_hash_after: 90da025455320fbb
  model_parameters_changed: true
  mean_completed_episode_steps: 10.81
  max_completed_episode_steps: 41
```

Read: the live6 runs are not starting collapsed to one action. The first
`model_hash_before` values also differ across checked runs, so the neural
network weights are not identical across the batch. These are sanity checks only
and not learning claims.

## Old Checkpoint Degeneracy Audit

Purpose: inspect discarded CurvyTron/Pong-era checkpoints for degenerate
behavior while live6 runs continue. Degenerate means the policy always or almost
always picks one action. The quick rule for now is:

```text
if one action is >= 95% of selected actions on random starts, mark the checkpoint degenerate
also record mean survival steps so action balance is not mistaken for competence
```

Existing CurvyTron eval support:

```text
src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py
```

It already records:

```text
episode.action_histogram
episode.steps
strict checkpoint load status
policy_could_act_in_real_env
```

Initial old checkpoint sets to audit:

```text
reference-v2:
training/lightzero-curvytron-visual-survival/curvytron-two-seat-selfplay-reference-v2-b32x128x128-u4-sim4-20260510/checkpoints/lightzero/iteration_{0..7}.pth.tar

live5 basic:
training/lightzero-curvytron-visual-survival/curvytron-two-seat-selfplay-live5-basic-survival001-b32-open-u8-sim4-20260510/checkpoints/lightzero/iteration_0.pth.tar
```

Next action: launch small parallel evals with random starts, low step cap, and
summary-only output; then record which old checkpoints are collapsed.

## Live9 GPU Two-Seat Relaunch - 2026-05-11

Why this exists: replace live7/live8 jobs with the optimizer speed-fixed GPU
two-seat path. The new path batches both active seats into one current
LightZero policy/search call per decision step, then steps both players
together. This is the current CurvyTron learning probe.

Common settings:

```text
compute: gpu-l4-t4
modal function CPU: 64
batch_size: 32
outer_iterations: 1000000
collect_steps_per_iteration: 256
learner_updates: 1
replay_scope: accumulated
learner_sample_size: 1024
max_replay_rows: 32768
max_ticks: 65536
checkpoint_every_iterations: 100
progress_every_iterations: 25
action_selection_mode: collect
```

Fresh live9 runs:

| run suffix | seed | updates | simulations | temp | eps | alive | dead | extra knob |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `base-u1-temp10-eps025-sim5` | 2001 | 1 | 5 | 1.0 | 0.25 | 0.01 | -1.0 | none |
| `base-u2-temp10-eps025-sim5` | 2002 | 2 | 5 | 1.0 | 0.25 | 0.01 | -1.0 | more learner pressure |
| `explore-u1-temp15-eps025-sim5` | 2003 | 1 | 5 | 1.5 | 0.25 | 0.01 | -1.0 | warmer policy sampling |
| `explore-u1-temp20-eps025-sim5` | 2004 | 1 | 5 | 2.0 | 0.25 | 0.01 | -1.0 | warmer policy sampling |
| `explore-u1-temp30-eps025-sim5` | 2005 | 1 | 5 | 3.0 | 0.25 | 0.01 | -1.0 | high policy sampling |
| `explore-u1-temp20-eps050-sim5` | 2006 | 1 | 5 | 2.0 | 0.50 | 0.01 | -1.0 | higher random action mix |
| `sim8-u1-temp20-eps025` | 2007 | 1 | 8 | 2.0 | 0.25 | 0.01 | -1.0 | deeper search |
| `sim8-u1-temp15-eps050` | 2008 | 1 | 8 | 1.5 | 0.50 | 0.01 | -1.0 | deeper search plus more random actions |
| `survival005-u1-temp20-eps025` | 2009 | 1 | 5 | 2.0 | 0.25 | 0.05 | -1.0 | stronger survival reward |
| `nodead-u1-temp20-eps025` | 2010 | 1 | 5 | 2.0 | 0.25 | 0.01 | 0.0 | no death penalty |
| `noop05warm-u1-temp20-eps025` | 2011 | 1 | 5 | 2.0 | 0.25 | 0.01 | -1.0 | no-op rises to 5% over 1000 iterations |
| `noop10warm-u1-temp20-eps025` | 2012 | 1 | 5 | 2.0 | 0.25 | 0.01 | -1.0 | no-op rises to 10% over 1000 iterations |
| `imgnoise005-u1-temp20-eps025` | 2013 | 1 | 5 | 2.0 | 0.25 | 0.01 | -1.0 | observation noise std 0.005 |
| `imgnoise010-u1-temp20-eps025` | 2014 | 1 | 5 | 2.0 | 0.25 | 0.01 | -1.0 | observation noise std 0.010 |
| `mixed-noop05-img005-u1-temp20-eps025` | 2015 | 1 | 5 | 2.0 | 0.25 | 0.01 | -1.0 | 5% no-op schedule plus 0.005 image noise |
| `wide-explore-noop10-img005-u1-temp30-eps050` | 2016 | 1 | 5 | 3.0 | 0.50 | 0.01 | -1.0 | high exploration plus 10% no-op schedule plus 0.005 image noise |

Run id pattern:

```text
curvytron-two-seat-selfplay-live9-<run suffix>-20260511
attempt id: live9-<run suffix>-20260511
```

Cleanup done after live9 spawned:

```text
stopped stale live7 apps:
ap-X20FtSC02H5rKprB70dSaW
ap-qtrRe2R7qiDQPsVjZv4TRK
ap-fMpRCCmj0bhOw7smZa4Zrr
ap-htiaX4O9JlFJsFnN86pqaH
ap-TUCpDMLbHsGgRoQrS3hKjq
ap-QDCCDiypQ0pqVABXGxFWVn
ap-ymwLpBXfkefc994ytrlnEJ
ap-WxivaWoTWPXopkitK6t7Eo
ap-fvsgCkTHP90EjxFCzbWPjc
ap-1zxZxJpbU8CNBP9dWd1CcQ
ap-RIq31BauLid0mK1mK9QnuS
ap-AduvFAaS6BizDdVEuByQKn

stopped stale live8 apps:
ap-ve1j9L3Aw5vrIqz9yLQQC4
ap-uRhOBYhH0QsFVDrrcs1Zxp
ap-Bm1bbUvieOGNuc8Na9x8d9
ap-HfMcggF74MUuLya8cYvCXu
ap-HeXQ35IV0LLFync75jCKKb
ap-T92D9NfyDiPAdXIXRETvgD
ap-b29Pl2P61zYEHRCftyfteC
ap-3I0gzjadMLjUK5gaRQmN04
```

Next check:

```text
Wait for progress rows. First read is health only:
- any problem rows?
- model_parameters_changed true?
- action counts not collapsed to one action?
- mean/max completed episode steps visible?

Learning read comes later:
- compare mean_completed_episode_steps over progress rows;
- compare checkpoints by survival-time eval, not score first.
```

## Live11 Corrected GPU Two-Seat Relaunch - 2026-05-11

Live9 was superseded. The first live10 relaunch tried to force 64 CPU cores on
the GPU Modal function. Modal rejected that because the GPU function currently
allows at most 40 CPU cores. The code now uses:

```text
CPU-only function: 64 CPU cores
GPU function: 40 CPU cores, L4/T4 GPU
```

Code sanity checked before relaunch:

```text
training collection still uses MuZeroPolicy.collect_mode.forward
GPU wrapper passes use_cuda=True
background jobs use Modal spawn
py_compile passes for the Modal wrapper
checkpoint default: 100 iterations
progress default: 100 iterations, overridden to 25 for live11
```

Fresh live11 runs:

```text
same 16 variants as live9
run id pattern: curvytron-two-seat-selfplay-live11-<run suffix>-20260511
attempt id: live11-<run suffix>-20260511
seeds: 4001 through 4016
progress_every_iterations: 25
checkpoint_every_iterations: 100
```

Live11 spawned count:

```text
16/16 spawned
```

Live9 app cleanup:

```text
stopped all 16 live9 app ids parsed from /tmp/live9_*.log
```

Next check remains the same: wait for progress rows, then inspect problem
counts, action histograms, model update flags, and completed episode steps.

## Live12 Player-Perspective + Episode-Return Fix - 2026-05-11

Why live12 exists:

```text
live11 was flat. Two setup bugs were found after the flat read:
1. both policy seats could receive ambiguous global visual frames;
2. survival return targets could be cut at outer-iteration boundaries.
```

Code fixes now in the live12 source copy:

```text
player observations:
  SourceStateGray64Stack4 remaps source player pixels into self/other values
  per controlled player before normalization; policy row shape stays [4,64,64].

to_play:
  LightZero collect/eval calls now pass player_id instead of -1.

survival targets:
  replay rows preserve episode_id;
  learner samples carry episode_id_batch and return_context_episode_id_batch;
  discounted survival return lookup groups by episode_id/env_row/player when available.
```

Local verification before launch:

```text
py_compile: passed for the touched training files
player-perspective probe: ok, reset delta > 0 and step delta > 0
local current-policy smoke: ok
split-iteration target proof: ok, returns [4, 3, 2, 1]
local installed-LightZero smoke: blocked because local LightZero policy setup is not available; Modal image has the installed LightZero path
```

Corrected long run:

```text
run_id: curvytron-two-seat-selfplay-live12-playerpersp-episodefix-clean-long-20260511
attempt_id: live12-playerpersp-episodefix-clean-long-20260511
Modal app id: ap-pzRnD0oXuFYb4N7yWzORA3
function_call_id: fc-01KRBMJJ0N2ZK3550F3TDVDBMC
compute: gpu-l4-t4
batch_size: 32
collect_steps_per_iteration: 128
outer_iterations: 100000
updates_per_iteration: 8
num_simulations: 4
replay_scope: accumulated
learner_sample_size: 512
max_replay_rows: 65536
max_ticks: 16384
decision_ms: 300
alive_reward: 0.01
dead_reward: -1.0
checkpoint_every_iterations: 100
progress_every_iterations: 25
```

Refs:

```text
progress_latest_ref:
training/lightzero-curvytron-visual-survival/curvytron-two-seat-selfplay-live12-playerpersp-episodefix-clean-long-20260511/attempts/live12-playerpersp-episodefix-clean-long-20260511/train/progress_latest.json

progress_ref:
training/lightzero-curvytron-visual-survival/curvytron-two-seat-selfplay-live12-playerpersp-episodefix-clean-long-20260511/attempts/live12-playerpersp-episodefix-clean-long-20260511/train/progress.jsonl

checkpoint_root_ref:
training/lightzero-curvytron-visual-survival/curvytron-two-seat-selfplay-live12-playerpersp-episodefix-clean-long-20260511/checkpoints/lightzero
```

Initial status check:

```text
progress_exists: true
latest event: start
latest checkpoint: iteration_0
```

Follow-up status, about 30 minutes later:

```text
Modal app: still alive with 1 task
before preemption: iteration 25 and iteration 50 were written
Modal event: worker preemption, function restarted with same input
after restart: latest progress row is a fresh start row at 2026-05-11T14:17:12Z
```

Partial preemption-before curve:

| iteration | mean completed episode steps | max completed episode steps | top action fraction | problem count |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 11.289 | 35 | about 0.38 | 0 |
| 50 | 11.194 | 45 | about 0.45 | 0 |

Read: the corrected job is mechanically healthy: batched policy rows, optimizer
updates, no problem rows, no action collapse. It has not shown a survival lift
yet. Because the worker was preempted, the live attempt needs another first
progress interval before reading the restarted curve.

Second follow-up status:

```text
Modal app: still alive with 1 task
latest row: iteration 50 after restart
latest mean completed episode steps: 11.044
latest max completed episode steps: 40
model_changed: true
problems: 0
top action fraction: 0.383
checkpoint count: 1, latest checkpoint iteration_0
```

Curve rows currently visible:

| pass | iteration | mean completed episode steps | max completed episode steps | top action fraction | problem count |
| --- | ---: | ---: | ---: | ---: | ---: |
| before preemption | 25 | 11.289 | 35 | 0.380 | 0 |
| before preemption | 50 | 11.194 | 45 | 0.447 | 0 |
| after restart | 25 | 11.334 | 40 | 0.353 | 0 |
| after restart | 50 | 11.044 | 40 | 0.383 | 0 |

Read: still mechanically healthy but no survival lift yet. Next useful read is
after checkpoint `iteration_100` exists.

First progress check after launch:

```text
iteration: 25
elapsed_sec: 687.184
total_steps_collected: 3200
total_replay_rows_collected: 204800
completed_episode_count: 402
mean_completed_episode_steps: 11.289
max_completed_episode_steps: 35
problem_count: 0
model_parameters_changed: true
top_action_fraction: 0.380
collapsed: false
latest checkpoint: iteration_0
```

Read: the corrected run is alive, writing progress, training the model, and not
collapsed. It has not run long enough for a learning claim. The next checkpoint
is expected at iteration 100.

Stale cleanup:

```text
Issued successful `modal app stop` calls for all 16 live11 Modal app ids. These
runs are stale because they used the older observation/target setup.
```
