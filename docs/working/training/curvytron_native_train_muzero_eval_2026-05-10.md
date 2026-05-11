# CurvyTron Native Train MuZero Inventory + Eval - 2026-05-10

Purpose: inventory the native LightZero `train_muzero` CurvyTron visual-survival
runs where the environment owns the opponent, then score the best completed
native checkpoint lane by survival steps. This note intentionally excludes the
custom two-seat adapter lane.

## Scope

Native lane here means:

```text
trainer_entrypoint: lzero.entry.train_muzero
env_id: CurvyZeroStackedDebugVisualSurvivalLightZero-v0
reward_schema_id: curvyzero_survival_time/v0
target reward: survival time
observation_shape: [4,64,64]
action_space_size: 3
opponent owned by env: fixed_straight or frozen checkpoint unless otherwise noted
current-policy two-seat self-play: not implemented in this lane
```

Do not read this as source-fidelity CurvyTron or current-policy self-play. The
main question is narrower: does the Pong-like native LightZero path show a
survival-time checkpoint curve when CurvyTron owns the fixed or frozen opponent
inside the env?

## Inventory

Sources checked:

```text
docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md
docs/working/training/curvytron_lightzero_visual_survival_training_worker_handoff_2026-05-10.md
Modal volume: curvyzero-runs/training/lightzero-curvytron-visual-survival/
```

Volume root listing command:

```text
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-curvytron-visual-survival
```

Native train_muzero runs with stable Modal volume roots:

| run_id | attempt_id | status | opponent | observation | checkpoint evidence | read |
|---|---:|---|---|---|---|---|
| `curvytron-visual-survival-debug-lz-s0` | `train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510` | completed | fixed straight | legacy non-player-aware | `iteration_0` through `iteration_73`, plus `ckpt_best` | first real native plumbing run; too small for quality |
| `curvytron-visual-survival-debug-lz-s16-livepublish-smoke` | `train-gpu-l4t4-survival-debug-livepublish-smoke-s16-20260510` | completed | fixed straight | legacy non-player-aware | `iteration_0` through `iteration_43`, plus `ckpt_best` | live checkpoint publish smoke |
| `curvytron-visual-survival-debug-lz-s40-wait-livepublish-32768` | `train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s40-20260510` | completed | fixed straight | legacy non-player-aware | through `iteration_262` | full waited native curve, flat |
| `curvytron-visual-survival-debug-lz-s41-wait-livepublish-sim16-32768` | `train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s41-sim16-20260510` | completed | fixed straight | legacy non-player-aware | through `iteration_153` | full waited native curve, flat |
| `curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536` | `train-gpu-l4t4-survival-debug-wait-livepublish-65536x256-s42-20260510` | completed | fixed straight | legacy non-player-aware | through `iteration_293` | full waited native curve, flat |
| `curvytron-visual-survival-player-aware-fixed-s100-131072` | `train-gpu-l4t4-player-aware-fixed-131072x512-s100-wait-20260510` | completed | fixed straight | player-aware v1 | through `iteration_520` | input bug fixed; curve flat |
| `curvytron-visual-survival-player-aware-fixed-s101-262144` | `train-gpu-l4t4-player-aware-fixed-262144x1024-s101-wait-20260510` | completed | fixed straight | player-aware v1 | through `iteration_1071` | best-looking existing native curve, but weak |
| `curvytron-visual-survival-player-aware-fixed-s102-sim8-131072` | `train-gpu-l4t4-player-aware-fixed-131072x512-s102-sim8-wait-20260510` | completed | fixed straight | player-aware v1 | through `iteration_540` | sim8 control, flat |
| `curvytron-bg-native-fixed-s103-262144` | `train-gpu-l4t4-bg-fixed-262144x1024-s103-detach-20260510` | completed | fixed straight | player-aware v1 | through `iteration_1057` | background fixed lane; modest late lift |
| `curvytron-bg-native-frozen-s42iter293-s104-131072` | `train-gpu-l4t4-bg-frozen-s42iter293-131072x512-s104-detach-20260510` | completed | frozen s42 `iteration_293` | player-aware v1 | through `iteration_518` | matched-opponent spikes, final falls back |

Named early runs checked but not found as stable checkpoint roots:

```text
curvytron-visual-survival-debug-lz-s13-hailmary131k: no checkpoint directory
curvytron-visual-survival-debug-lz-s14-sim16-32k: no checkpoint directory
s10/s11/s12: docs mention them as in-flight wider-wave launches, but no stable
  volume roots were present under training/lightzero-curvytron-visual-survival/
```

Verification commands for missing s13/s14:

```text
uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s13-hailmary131k/checkpoints/lightzero

uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s14-sim16-32k/checkpoints/lightzero
```

Both returned `No such file or directory`.

## Existing Curves

Legacy waited runs, 8 eval starts, cap 1024:

| run | checkpoints | best mean steps | latest mean steps | verdict |
|---|---:|---:|---:|---|
| s40 | 8 | 218.38 at `iteration_200` | 199.38 at `iteration_262` | no upward trend |
| s41 | 8 | 219.00 at `iteration_96` | 211.38 at `iteration_153` | no upward trend |
| s42 | 8 | 199.38 at `iteration_80` | 194.88 at `iteration_293` | no upward trend |

Player-aware fixed-straight curves from the experiment log, 64 random starts,
cap 1024:

| run | checkpoints | mean steps curve | read |
|---|---:|---|---|
| s100 | 6 | `178.531, 164.625, 170.172, 160.875, 160.375, 171.906` | lower than initial |
| s101 | 6 | `154.812, 158.391, 171.141, 166.047, 170.141, 171.375` | best-looking small lift |
| s102 | 6 | `178.531, 166.984, 177.219, 178.625, 177.422, 174.719` | flat |

The best existing completed native curve is s101 by late-vs-initial movement:
`154.812 -> 171.375` mean steps over 64 starts. That is a small lift, not a
robust learning result.

## Fresh Eval

I reran the best completed native candidate, s101, over 8 spread checkpoints
and an 8-seed random panel.

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id curvytron-visual-survival-player-aware-fixed-s101-262144 \
  --attempt-id train-gpu-l4t4-player-aware-fixed-262144x1024-s101-wait-20260510 \
  --eval-id native_inventory_s101_curve_rand8_8pt_20260510 \
  --selected-iterations 0,160,320,480,640,800,960,1071 \
  --eval-seed-count 8 \
  --eval-seed-rng-seed 20260510 \
  --max-eval-steps 1024 \
  --source-max-steps 1024 \
  --num-simulations 4 \
  --batch-size 32 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

Random seed panel:

```text
eval_seed_sampler_seed: 20260510
eval_seeds: 1093491367,1646752993,983581866,1481646630,264468913,612598383,248211689,31836349
```

Manifest:

```text
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-player-aware-fixed-s101-262144/attempts/train-gpu-l4t4-player-aware-fixed-262144x1024-s101-wait-20260510/eval/native_inventory_s101_curve_rand8_8pt_20260510/manifest_steps1024_seedsn8_fea64279a80b_20260510T220139Z.json
```

Fresh s101 survival curve, mean over 8 starts:

| checkpoint | seeds | mean_steps | median | min | max | ok |
|---|---:|---:|---:|---:|---:|---:|
| `iteration_0` | 8 | 195.25 | 189.5 | 107 | 312 | 8/8 |
| `iteration_160` | 8 | 181.25 | 169.5 | 77 | 312 | 8/8 |
| `iteration_320` | 8 | 163.25 | 152.5 | 76 | 312 | 8/8 |
| `iteration_480` | 8 | 140.75 | 129.0 | 40 | 312 | 8/8 |
| `iteration_640` | 8 | 158.00 | 160.5 | 65 | 265 | 8/8 |
| `iteration_800` | 8 | 175.00 | 137.0 | 100 | 312 | 8/8 |
| `iteration_960` | 8 | 149.75 | 149.5 | 45 | 312 | 8/8 |
| `iteration_1071` | 8 | 165.50 | 133.0 | 62 | 312 | 8/8 |

## Verdict

Native LightZero `train_muzero` with the CurvyTron env owning the opponent is
mechanically viable: runs complete, checkpoints publish, strict checkpoint
loads pass, and survival evals execute over random seed panels.

It now has weak positive signs, but not a robust pass. Fixed s103 improves late
without any capped episodes. Frozen s104 shows matched-opponent spikes and a few
caps, but the final checkpoint falls back. Keep evals as parallel survival-step
curves over reproducible random starts, not raw win/loss or single-seed stories.

## 2026-05-11 Long Native Eval Results

Main metric: steps survived. Score is secondary for this lane. A better curve
means checkpoints survive longer across the same random-start panel, preferably
with more 1024-step caps.

All rows below use 32 random starts and a 1024-step cap.

### Fixed Straight s203

```text
run_id: curvytron-bg-native-fixed-s203-1048576
attempt_id: train-gpu-l4t4-bg-fixed-1048576x4096-s203-save16-20260510
opponent: fixed_straight
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

### Fixed Straight s205

```text
run_id: curvytron-bg-native-fixed-s205-1048576
attempt_id: train-gpu-l4t4-bg-fixed-1048576x4096-s205-detach-20260510
opponent: fixed_straight
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

### Frozen s47 iteration_200 Opponent, s204

```text
run_id: curvytron-bg-native-frozen-s47iter200-s204-1048576
attempt_id: train-gpu-l4t4-bg-frozen-s47iter200-1048576x4096-s204-save8-20260510
opponent: frozen s47 iteration_200
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

Read: real matched-opponent survival signal, but unstable. Check it against a
broader opponent panel before treating it as general learning.

### Frozen s92 iteration_434 Opponent, s206

```text
run_id: curvytron-bg-native-frozen-s92iter434-s206-524288
attempt_id: train-gpu-l4t4-bg-frozen-s92iter434-524288x2048-s206-detach-20260510
opponent: frozen s92 iteration_434
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
can show signal, but they may overfit to one opponent. The next eval should use
a small opponent panel for the best checkpoints: `fixed_straight`, frozen s47
`iteration_200`, and frozen s92 `iteration_434`.

## 2026-05-10 Background Fixed Run Eval

Run:

```text
run_id: curvytron-bg-native-fixed-s103-262144
attempt_id: train-gpu-l4t4-bg-fixed-262144x1024-s103-detach-20260510
eval_id: bg_native_fixed_s103_curve_fixedstraight_rand32_20260510
eval seed count: 32
eval seed sampler seed: 20260510
cap: 1024 steps
opponent: fixed_straight
```

Manifest:

```text
training/lightzero-curvytron-visual-survival/curvytron-bg-native-fixed-s103-262144/attempts/train-gpu-l4t4-bg-fixed-262144x1024-s103-detach-20260510/eval/bg_native_fixed_s103_curve_fixedstraight_rand32_20260510/manifest_steps1024_seedsn32_b39991066252_20260510T222534Z.json
```

Survival curve:

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

Read: this is a modest survival improvement, not a solved result. It is the
first fixed-opponent background curve with a clear late lift on 32 starts, but
the max remains far below the 1024 cap.

## 2026-05-10 Background Frozen Run Eval

Run:

```text
run_id: curvytron-bg-native-frozen-s42iter293-s104-131072
attempt_id: train-gpu-l4t4-bg-frozen-s42iter293-131072x512-s104-detach-20260510
eval_id: bg_native_frozen_s104_curve_matched_s42iter293_rand32_20260510
eval seed count: 32
eval seed sampler seed: 20260510
cap: 1024 steps
opponent: frozen_lightzero_checkpoint
opponent_checkpoint_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar
opponent_snapshot_ref: curvytron_visual_survival_s42_iteration_293
```

Manifest:

```text
training/lightzero-curvytron-visual-survival/curvytron-bg-native-frozen-s42iter293-s104-131072/attempts/train-gpu-l4t4-bg-frozen-s42iter293-131072x512-s104-detach-20260510/eval/bg_native_frozen_s104_curve_matched_s42iter293_rand32_20260510/manifest_steps1024_seedsn32_b39991066252_20260510T222924Z.json
```

Survival curve:

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

Read: matched-opponent spikes are real on this panel, but unstable. The last
checkpoint is only a little above iteration 0 and has no caps.

## 2026-05-10 Long Run Checkpoint Notes

Fresh long native runs have completed training artifacts and long evals:

| run | opponent | save cadence | latest checkpoint | completion | completed long eval points |
|---|---|---:|---:|---|---|
| `curvytron-bg-native-fixed-s203-1048576` | fixed_straight | 16 | `iteration_4106` | complete; `summary.json` present | `0,512,1024,1536,2048,2560,3072,3584,4096,4106` |
| `curvytron-bg-native-frozen-s47iter200-s204-1048576` | frozen `s47 iteration_200` | 8 | `iteration_4105` | complete; `summary.json` present | `0,512,1024,1536,2048,2560,3072,3584,4096,4105` |
| `curvytron-bg-native-fixed-s205-1048576` | fixed_straight | 16 | `iteration_4125` | complete; `summary.json` present | `0,512,1024,1536,2048,2560,3072,3584,4096,4125` |
| `curvytron-bg-native-frozen-s92iter434-s206-524288` | frozen `s92 iteration_434` | 8 | `iteration_2099` | complete; `summary.json` present | `0,256,512,768,1024,1280,1536,1792,2048,2099` |

Modal volume ranges:

```text
s203 fixed: 258 iteration checkpoints, 0..4106, cadence 16 plus final
s204 frozen s47iter200: 515 iteration checkpoints, 0..4105, cadence 8 plus final
s205 fixed: 259 iteration checkpoints, 0..4125, cadence 16 plus final
s206 frozen s92iter434: 264 iteration checkpoints, 0..2099, cadence 8 plus final
```

The wrapper default is still to save every learner iteration. Long jobs may
override that upward, but every long launch must state `--save-ckpt-after-iter`
explicitly so checkpoint curves are not dependent on memory.

No pytest was run.
