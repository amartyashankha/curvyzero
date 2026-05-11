# CurvyTron Visual Survival LightZero Train - 2026-05-10

Purpose: first real LightZero `train_muzero` run for CurvyTron stacked
debug-visual survival. This is a debug-fidelity training plumbing run, not a
source-fidelity visual CurvyTron or policy-quality claim.

Signal rule: CurvyTron reward and eval signal are steps survived. Use
reproducible random eval panels and record the sampler seed plus exact seed
list; fixed seed panels are for replay/debug, not the main habit.

## Code Path

Trainer wrapper:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Env action telemetry was added to:

```text
src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_smoke.py
```

Active env:

```text
env.type: curvyzero_stacked_debug_visual_survival_lightzero
env.import_names: ["curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env"]
env_id: CurvyZeroStackedDebugVisualSurvivalLightZero-v0
reward_schema_id: curvyzero_survival_time/v0
observation_schema_id: curvyzero_stacked_debug_occupancy_gray64_player_aware_survival_time/v1
frame_stack_owner: curvyzero_wrapper_local_debug_frame_stack
debug_fidelity_only: true
source_fidelity_claim: none
```

Player-aware observation correction:

```text
old checkpoints: curvyzero_stacked_debug_occupancy_gray64_survival_time/v0
old raw heads: curvyzero_debug_occupancy_gray64/v0 global anonymous live avatars
new player-aware two-seat/checkpoint-opponent path: curvyzero_stacked_debug_occupancy_gray64_player_aware_survival_time/v1
new raw heads: curvyzero_debug_occupancy_gray64_player_aware/v1
controlled avatar pixel: 255
other live avatar pixel: 220
world body pixel: 160
```

The v0 visual checkpoints used the same live-avatar mark for ego and opponent.
That is acceptable only as a legacy plumbing artifact. For any future two-seat
current-policy path, frozen checkpoint opponents, and any run where the same
model may occupy different seats, use the player-aware path so the ego stack is
rendered with `ego_player_id` as controlled and the frozen opponent stack is
rendered with `opponent_player_id` as controlled.

Important stack fix:

```text
env observation_shape: [4,64,64]
policy.model.observation_shape: [4,64,64]
env.frame_stack_num: 1
policy.model.frame_stack_num: 1
policy.model.image_channel: 4
```

The failed pre-fix train attempt proved why this matters:

```text
attempt_id: train-gpu-l4t4-survival-debug-4096x32-wait-20260510
called_train_muzero: true
status: failed
problem: expected input[1,16,64,64] to have 4 channels
cause: LightZero stacked the wrapper-stacked [4,64,64] observation again
```

## Verification

Commands:

```text
uv run python -m py_compile \
  src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_smoke.py \
  src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode dry \
  --compute cpu \
  --seed 0 \
  --run-id curvytron-visual-survival-debug-lz-s0 \
  --attempt-id dry-config-visual-survival-debug-stackfix-20260510 \
  --max-env-step 4096 \
  --max-train-iter 32 \
  --source-max-steps 256 \
  --num-simulations 8 \
  --batch-size 16 \
  --save-ckpt-after-iter 1
```

Dry result:

```text
ok: true
called_train_muzero: false
summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/attempts/dry-config-visual-survival-debug-stackfix-20260510/train/summary.json
```

## Successful Train Run

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 0 \
  --run-id curvytron-visual-survival-debug-lz-s0 \
  --attempt-id train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510 \
  --max-env-step 4096 \
  --max-train-iter 32 \
  --source-max-steps 256 \
  --num-simulations 8 \
  --batch-size 16 \
  --save-ckpt-after-iter 1 \
  --wait-for-train
```

Result:

```text
Modal app: ap-zXhm4AFaq8MI78MDtRNdjd
ok: true
called_train_muzero: true
return_type: MuZeroPolicy
compute: gpu-l4-t4
remote elapsed: 74.799413 sec
train_muzero elapsed: 47.568433 sec
problems: []
final_rewards logged: [84.0, 80.0]
summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/attempts/train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510/train/summary.json
```

Action observability:

```text
action telemetry rows: 461
ego_action_histogram: {"0": 226, "1": 108, "2": 127}
opponent_action_histogram: fixed straight through opponent_action_id=1
done_count: 4
reward_sum: 459.0
reward_mean: 0.9956616052060737
collapse_warning: null
action_observability_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/attempts/train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510/train/action_observability.json
```

Checkpoints:

```text
mirrored checkpoint count: 75
checkpoint root: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/checkpoints/lightzero/
includes:
  ckpt_best.pth.tar
  iteration_0.pth.tar
  ...
  iteration_73.pth.tar
```

Note: the command requested `max_train_iter=32`, but installed LightZero still
saved checkpoints through `iteration_73` before the `max_env_step=4096` bound
ended the job. Treat `max_env_step` as the effective bound for this run until
the LightZero iteration cap behavior is audited.

## Claims

Allowed:

```text
Real installed LightZero train_muzero ran against the CurvyTron stacked debug
visual survival env with survival-only reward, frequent checkpoints, and
env-side action histograms.
```

Not allowed:

```text
No source-fidelity visual claim.
No full multiplayer self-play claim.
No policy-quality claim.
No Pong-solved claim.
No scalar-only evidence should be used to describe this run.
```

## Two-Seat Iterative Smoke

Run:

```text
run_id: curvytron-two-seat-iterative-b8-s7-4x32-u2-sim2
status: passed
collect_update_rounds: 4
replay_rows: 2048
checkpoints: 5
```

Eval over 32 random starts:

```text
iteration_0: mean_steps 181.28125
iteration_1: mean_steps 174.0625
iteration_2: mean_steps 170.71875
iteration_3: mean_steps 174.125
iteration_4: mean_steps 170.71875
```

Claim: the iterative two-seat collect/update/checkpoint/eval plumbing passed.
Non-claim: the checkpoint curve does not show a learning signal yet.

Current blockers before scaling more two-seat runs:

```text
1. target_value is immediate reward, not discounted survival return; target fix
   is in progress.
2. learner batch sizing is suspect: the LightZero profile hard-sets
   policy.batch_size=2 and _learn_mode_batches slices samples to that size, so
   big CurvyTron collect phases may have updated on only 2 replay rows.
```

Do not read flat two-seat curves as a policy-capacity verdict until both the
value target and learner batch-size path are fixed and rerun.

## Urgent Follow-Up Launches

Current wider-wave runs already in flight before this note:

```text
s10/s11: 32768x128
s12: 65536x256
```

Launched two additional background runs, avoiding those ids:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 14 \
  --run-id curvytron-visual-survival-debug-lz-s14-sim16-32k \
  --attempt-id train-gpu-l4t4-survival-debug-32768x128-s14-sim16-20260510 \
  --max-env-step 32768 \
  --max-train-iter 128 \
  --source-max-steps 1024 \
  --num-simulations 16 \
  --batch-size 32 \
  --save-ckpt-after-iter 4
```

```text
Modal app: ap-nYLT88ZGlSxGNxo8CywJKA
function_call_id: fc-01KR95MRWF547QDNDZAC4XWMYG
summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s14-sim16-32k/attempts/train-gpu-l4t4-survival-debug-32768x128-s14-sim16-20260510/train/summary.json
action_observability_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s14-sim16-32k/attempts/train-gpu-l4t4-survival-debug-32768x128-s14-sim16-20260510/train/action_observability.json
```

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 13 \
  --run-id curvytron-visual-survival-debug-lz-s13-hailmary131k \
  --attempt-id train-gpu-l4t4-survival-debug-131072x512-s13-hailmary-20260510 \
  --max-env-step 131072 \
  --max-train-iter 512 \
  --source-max-steps 2048 \
  --num-simulations 8 \
  --batch-size 32 \
  --save-ckpt-after-iter 8
```

```text
Modal app: ap-grD5fS18CXOzi5g0d6KrdA
function_call_id: fc-01KR95MS06C0W8HNHMRSSZYRZY
summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s13-hailmary131k/attempts/train-gpu-l4t4-survival-debug-131072x512-s13-hailmary-20260510/train/summary.json
action_observability_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s13-hailmary131k/attempts/train-gpu-l4t4-survival-debug-131072x512-s13-hailmary-20260510/train/action_observability.json
```

No quality claim. Next check is summary/action telemetry: `ok`,
`called_train_muzero`, checkpoint count/root, `[4,64,64]` surface, survival-only
reward, and ego action collapse.

Important trainer scope: this CurvyTron visual survival trainer is learned ego
against a fixed straight opponent. It is not multiplayer self-play.

## Live Checkpoint Publish Fix

Issue found during long Hail Mary runs: the trainer mirrored LightZero
checkpoints into the stable run checkpoint root only after `train_muzero`
returned, so live eval could not see checkpoints while runs were active.

Fix:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

The trainer now patches LightZero `BaseLearner.save_checkpoint` during train
runs. After each save it rescans `lightzero_exp`, mirrors checkpoints into:

```text
training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero/
```

and writes/commits:

```text
training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/live_checkpoint_publish.json
```

Smoke command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 16 \
  --run-id curvytron-visual-survival-debug-lz-s16-livepublish-smoke \
  --attempt-id train-gpu-l4t4-survival-debug-livepublish-smoke-s16-20260510 \
  --max-env-step 2048 \
  --max-train-iter 16 \
  --source-max-steps 256 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --n-episode 1 \
  --num-simulations 4 \
  --batch-size 8 \
  --save-ckpt-after-iter 1 \
  --wait-for-train
```

Result:

```text
Modal app: ap-SbdvL5LhgnPZ6hTGdIFTYK
ok: true
status: completed
run_id: curvytron-visual-survival-debug-lz-s16-livepublish-smoke
attempt_id: train-gpu-l4t4-survival-debug-livepublish-smoke-s16-20260510
checkpoint_mirror count: 45
copied_now_count: 44
latest visible iteration: iteration_43.pth.tar
action rows: 346
ego_action_histogram: {"0": 101, "1": 121, "2": 124}
opponent_action_histogram: fixed straight through opponent_action_id=1
```

Volume visibility proof:

```text
modal volume ls curvyzero-runs training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s16-livepublish-smoke/checkpoints/lightzero
```

The stable checkpoint root included both:

```text
iteration_0.pth.tar
iteration_43.pth.tar
```

## CurvyTron Checkpoint Eval Harness

Added a minimal Modal eval harness:

```text
src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py
```

What it does:

```text
loads a mirrored LightZero MuZero checkpoint
strict-loads it into the same CurvyTron stacked debug visual survival policy config
runs the registered stacked debug visual survival env
records survival-time reward only
writes one JSON artifact per checkpoint/seed plus a manifest table
```

Output root:

```text
training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/eval/<eval_id>/
```

Manifest table fields:

```text
survival_aggregate_table:
  checkpoint
  seeds
  ok
  mean_steps
  median_steps
  min_steps
  max_steps
  mean_score

per-seed table:
checkpoint_label
checkpoint_ref
seed
steps_survived
total_reward
action_histogram
terminal_reason
cap
strict_load
elapsed_seconds
artifact_ref
```

Curve command shape:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id <run_id> \
  --attempt-id <attempt_id> \
  --selected-iterations 0,100,200 \
  --eval-seed-count 8 \
  --eval-seed-rng-seed 20260510 \
  --max-eval-steps 256 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

The default eval panel is also `8` sampled seeds from sampler seed `20260510`.
Use explicit `--eval-seeds` when replaying a recorded panel exactly. The stdout
summary now leads with checkpoint-level survival aggregates, then prints the
per-seed rows. Score/reward remains secondary.

Use explicit refs when the desired files are not named `iteration_<n>.pth.tar`
under the run checkpoint directory:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id <run_id> \
  --attempt-id <attempt_id> \
  --checkpoint-refs training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero/iteration_0.pth.tar,training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero/ckpt_best.pth.tar \
  --seed <seed> \
  --max-eval-steps 256 \
  --parallel \
  --summary-only
```

Small launch completed:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id curvytron-visual-survival-debug-lz-s0 \
  --attempt-id train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510 \
  --selected-iterations 0 \
  --seed 0 \
  --max-eval-steps 4 \
  --step-detail-limit 2 \
  --summary-only \
  --quiet-framework-logs
```

Result:

```text
Modal app: ap-L23wIVAw2BWG1tCSq9Z627
manifest_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/attempts/train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510/eval/checkpoint_curve/manifest_steps4_seeds0_20260510T185925Z.json
checkpoint: iteration_0
seed: 0
strict_load: true
steps_survived: 4
total_reward: 4.0
action_histogram: {"0": 4}
terminal_reason: cap
cap: 4
elapsed_seconds: 10.301248
```

Live-publish eval curve completed:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id curvytron-visual-survival-debug-lz-s16-livepublish-smoke \
  --attempt-id train-gpu-l4t4-survival-debug-livepublish-smoke-s16-20260510 \
  --selected-iterations 0,43 \
  --eval-seeds 16,17 \
  --max-eval-steps 256 \
  --source-max-steps 256 \
  --num-simulations 4 \
  --batch-size 8 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

Result:

```text
Modal app: ap-haGJfHip0F9h4dIRm1x3em
manifest_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s16-livepublish-smoke/attempts/train-gpu-l4t4-survival-debug-livepublish-smoke-s16-20260510/eval/checkpoint_curve/manifest_steps256_seeds16-17_20260510T190933Z.json
iteration_0 seed 16: strict_load=true, steps_survived=173, terminal_reason=survivor_win, action_histogram={"1": 173}
iteration_43 seed 16: strict_load=true, steps_survived=173, terminal_reason=survivor_win, action_histogram={"0": 146, "1": 27}
iteration_0 seed 17: strict_load=true, steps_survived=164, terminal_reason=survivor_win, action_histogram={"1": 165}
iteration_43 seed 17: strict_load=true, steps_survived=180, terminal_reason=survivor_win, action_histogram={"0": 180}
```

No pytest was run. Compile check only:

```text
uv run python -m py_compile \
  src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py \
  src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py
```

## Eval Harness Fix - Opponent Defaults

s44/s45 eval jobs exposed a real eval-lane bug: all jobs failed before env
reset because `_build_visual_survival_configs` now requires explicit opponent
arguments:

```text
opponent_policy_kind
opponent_checkpoint
opponent_snapshot_ref
opponent_checkpoint_state_key
```

Fix in:

```text
src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py
```

The eval harness now passes the default fixed-straight opponent settings:

```text
opponent_policy_kind=DEFAULT_OPPONENT_POLICY_KIND
opponent_checkpoint=None
opponent_snapshot_ref=None
opponent_checkpoint_state_key=None
```

This restores the intended learned-ego-vs-fixed-straight eval surface. It is
not a self-play claim. Re-run s44/s45 evals with the survival aggregate table
before reading them as curves.

## s30 Live-Publish Training Launch

Purpose: first longer CurvyTron debug visual survival train after live
checkpoint publishing landed. This is still learned ego against a fixed
straight opponent, not self-play.

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute gpu-l4-t4 \
  --seed 30 \
  --run-id curvytron-visual-survival-debug-lz-s30-livepublish-32768 \
  --attempt-id train-gpu-l4t4-survival-debug-livepublish-32768x128-s30-20260510 \
  --max-env-step 32768 \
  --max-train-iter 128 \
  --source-max-steps 1024 \
  --collector-env-num 4 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 2 \
  --n-episode 4 \
  --num-simulations 8 \
  --batch-size 32 \
  --save-ckpt-after-iter 4
```

Launch result:

```text
Modal app: ap-o06ImkzrSwI8SGU1hsrvw9
function_call_id: fc-01KR9MZBQ4FN83T39QFE3R7N3Y
summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s30-livepublish-32768/attempts/train-gpu-l4t4-survival-debug-livepublish-32768x128-s30-20260510/train/summary.json
action_observability_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s30-livepublish-32768/attempts/train-gpu-l4t4-survival-debug-livepublish-32768x128-s30-20260510/train/action_observability.json
```

Next check: verify that
`training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s30-livepublish-32768/checkpoints/lightzero/`
starts showing live `iteration_<n>.pth.tar` files, then eval `iteration_0`
against the newest visible checkpoint.

## s31/s32 Parallel Live-Publish Launches

Purpose: add two more CurvyTron debug visual survival curves so the first
CurvyTron read is not one lucky run. These are still learned ego against fixed
straight opponent, not self-play.

```text
s31:
run_id: curvytron-visual-survival-debug-lz-s31-livepublish-sim16-32768
attempt_id: train-gpu-l4t4-survival-debug-livepublish-32768x128-s31-sim16-20260510
Modal app: ap-F8iLu3T0DDCqyJ04S61YHB
function_call_id: fc-01KR9N7R9WGEV131Y91SVJ25RB
config: 32768 max_env_step, 128 max_train_iter, source_max_steps 1024,
  collector_env_num 4, num_simulations 16, batch_size 32,
  save_ckpt_after_iter 4

s32:
run_id: curvytron-visual-survival-debug-lz-s32-livepublish-65536
attempt_id: train-gpu-l4t4-survival-debug-livepublish-65536x256-s32-20260510
Modal app: ap-UmUhN0S33eoP3y90eOFAD2
function_call_id: fc-01KR9N7RA5XJH1GY16EW2VTN65
config: 65536 max_env_step, 256 max_train_iter, source_max_steps 1024,
  collector_env_num 4, num_simulations 8, batch_size 32,
  save_ckpt_after_iter 8
```

Next check: wait for live checkpoint roots, then eval `iteration_0` plus the
newest visible checkpoint for s30, s31, and s32.

Update: s30/s31/s32 were launched as background `.spawn` calls from ephemeral
`modal run` apps. Those function calls terminated when the local entrypoint app
stopped, so no volume roots appeared. Use `--wait-for-train` for current long
training launches, or deploy the app before relying on true background jobs.

## s40/s41/s42 Waited Replacement Launches

These runs keep the Modal app alive with `--wait-for-train`, so live checkpoint
publishing can actually happen.

```text
s40:
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train --compute gpu-l4-t4 --seed 40 \
  --run-id curvytron-visual-survival-debug-lz-s40-wait-livepublish-32768 \
  --attempt-id train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s40-20260510 \
  --max-env-step 32768 --max-train-iter 128 --source-max-steps 1024 \
  --collector-env-num 4 --evaluator-env-num 1 --n-evaluator-episode 2 \
  --n-episode 4 --num-simulations 8 --batch-size 32 \
  --save-ckpt-after-iter 4 --wait-for-train
Modal app: ap-mt6JBIqU0Qnru3XEgIKO14

s41:
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train --compute gpu-l4-t4 --seed 41 \
  --run-id curvytron-visual-survival-debug-lz-s41-wait-livepublish-sim16-32768 \
  --attempt-id train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s41-sim16-20260510 \
  --max-env-step 32768 --max-train-iter 128 --source-max-steps 1024 \
  --collector-env-num 4 --evaluator-env-num 1 --n-evaluator-episode 2 \
  --n-episode 4 --num-simulations 16 --batch-size 32 \
  --save-ckpt-after-iter 4 --wait-for-train
Modal app: ap-IYivO65mexWBf30BtTrS5z

s42:
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train --compute gpu-l4-t4 --seed 42 \
  --run-id curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536 \
  --attempt-id train-gpu-l4t4-survival-debug-wait-livepublish-65536x256-s42-20260510 \
  --max-env-step 65536 --max-train-iter 256 --source-max-steps 1024 \
  --collector-env-num 4 --evaluator-env-num 1 --n-evaluator-episode 2 \
  --n-episode 4 --num-simulations 8 --batch-size 32 \
  --save-ckpt-after-iter 8 --wait-for-train
Modal app: ap-NrAlFxNrpBW5dhw5tM5IF2
```

Result: all three waited runs completed and published checkpoints into the
stable run roots. These are still learned ego against a fixed straight
opponent, not self-play.

```text
s40:
  latest visible checkpoint: iteration_262.pth.tar
  action histogram from train summary: {"0": 554, "1": 529, "2": 343}
  done_count: 12

s41:
  latest visible checkpoint: iteration_153.pth.tar
  action histogram from train summary: {"0": 270, "1": 273, "2": 281}
  done_count: 6

s42:
  latest visible checkpoint: iteration_293.pth.tar
  action histogram from train summary: {"0": 765, "1": 447, "2": 345}
  done_count: 12
```

Full-curve eval command shape:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id <run_id> \
  --attempt-id <attempt_id> \
  --selected-iterations <8 spread checkpoints> \
  --eval-seeds 1013,2029,3037,4049,5051,6067,7079,8093 \
  --max-eval-steps 1024 \
  --source-max-steps 1024 \
  --num-simulations <8 or 16> \
  --batch-size 32 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

Eval artifacts:

```text
s40 manifest:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s40-wait-livepublish-32768/attempts/train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s40-20260510/eval/checkpoint_curve/manifest_steps1024_seeds1013-2029-3037-4049-5051-6067-7079-8093_20260510T192815Z.json

s41 manifest:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s41-wait-livepublish-sim16-32768/attempts/train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s41-sim16-20260510/eval/checkpoint_curve/manifest_steps1024_seeds1013-2029-3037-4049-5051-6067-7079-8093_20260510T192815Z.json

s42 manifest:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/attempts/train-gpu-l4t4-survival-debug-wait-livepublish-65536x256-s42-20260510/eval/checkpoint_curve/manifest_steps1024_seeds1013-2029-3037-4049-5051-6067-7079-8093_20260510T192815Z.json
```

Survival-step curve, mean over 8 eval starts:

```text
s40, 8 search simulations, cap 1024
iteration  mean_steps  median  min  max
0          199.00      222.5   40   316
40         156.50      150.5   126  227
80         181.25      178.5   40   291
120        162.50      156.5   107  227
160        149.75      159.5   54   290
200        218.38      233.5   126  299
240        199.75      209.5   54   316
262        199.38      173.0   86   316

s41, 16 search simulations, cap 1024
iteration  mean_steps  median  min  max
0          206.25      222.5   54   316
24         218.62      222.5   126  313
48         194.50      179.5   57   312
72         166.12      159.5   57   290
96         219.00      213.5   126  316
120        197.00      189.5   126  283
144        174.25      159.0   105  291
153        211.38      215.0   105  316

s42, 8 search simulations, cap 1024
iteration  mean_steps  median  min  max
0          199.00      222.5   40   316
40         198.75      222.5   40   316
80         199.38      222.5   40   316
120        181.50      187.5   62   291
160        162.75      158.5   69   291
200        179.88      164.5   48   316
240        199.00      222.5   40   316
293        194.88      220.5   41   316
```

Plain read: these curves do not show a clean climb. They prove that the
waited Modal pattern, checkpoint publishing, strict checkpoint load, and
parallel eval path work. They do not prove useful CurvyTron learning. More
fixed-straight-opponent runs are lower value than wiring the frozen-checkpoint
opponent path and then moving toward current-policy self-play.

## s40/s41/s42 Quick Rand3 Survival Curves

Follow-up check: list visible Modal Volume checkpoints first, then run compact
survival-step eval curves over 8 checkpoint points per run. No pytest was run.

Visible checkpoint iterations from:

```text
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero
```

```text
s40 visible: iteration_0, iteration_4..iteration_260 every 4, plus iteration_262
s41 visible: iteration_0, iteration_4..iteration_152 every 4, plus iteration_153
s42 visible: iteration_0, iteration_8..iteration_288 every 8, plus iteration_293
```

Eval seed panel was deterministic and random-looking:

```text
uv run python -c 'import random; rng=random.Random(20260510); print(",".join(map(str, rng.sample(range(0, 2**31-1), 3))))'
seeds: 546745683,1247268015,823376496
```

Commands:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id curvytron-visual-survival-debug-lz-s40-wait-livepublish-32768 \
  --attempt-id train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s40-20260510 \
  --eval-id checkpoint_curve_rand3_8pt_20260510 \
  --selected-iterations 0,40,80,120,160,200,240,262 \
  --seed 546745683 \
  --eval-seeds 546745683,1247268015,823376496 \
  --max-eval-steps 256 \
  --parallel --summary-only --quiet-framework-logs

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id curvytron-visual-survival-debug-lz-s41-wait-livepublish-sim16-32768 \
  --attempt-id train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s41-sim16-20260510 \
  --eval-id checkpoint_curve_rand3_8pt_20260510 \
  --selected-iterations 0,24,48,72,96,120,144,153 \
  --seed 546745683 \
  --eval-seeds 546745683,1247268015,823376496 \
  --max-eval-steps 256 \
  --parallel --summary-only --quiet-framework-logs

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536 \
  --attempt-id train-gpu-l4t4-survival-debug-wait-livepublish-65536x256-s42-20260510 \
  --eval-id checkpoint_curve_rand3_8pt_20260510 \
  --selected-iterations 0,40,80,120,160,200,240,293 \
  --seed 546745683 \
  --eval-seeds 546745683,1247268015,823376496 \
  --max-eval-steps 256 \
  --parallel --summary-only --quiet-framework-logs
```

Manifests:

```text
s40:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s40-wait-livepublish-32768/attempts/train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s40-20260510/eval/checkpoint_curve_rand3_8pt_20260510/manifest_steps256_seeds546745683-1247268015-823376496_20260510T192818Z.json

s41:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s41-wait-livepublish-sim16-32768/attempts/train-gpu-l4t4-survival-debug-wait-livepublish-32768x128-s41-sim16-20260510/eval/checkpoint_curve_rand3_8pt_20260510/manifest_steps256_seeds546745683-1247268015-823376496_20260510T192927Z.json

s42:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/attempts/train-gpu-l4t4-survival-debug-wait-livepublish-65536x256-s42-20260510/eval/checkpoint_curve_rand3_8pt_20260510/manifest_steps256_seeds546745683-1247268015-823376496_20260510T193033Z.json
```

Survival steps, mean over 3 eval starts, cap 256:

```text
s40
iteration  mean_steps  raw_steps
0          194.333     256,132,195
40         182.667     256,97,195
80         194.333     256,132,195
120        185.333     256,132,168
160        194.333     256,132,195
200        173.000     226,125,168
240        194.333     256,132,195
262        180.333     250,96,195

s41
iteration  mean_steps  raw_steps
0          194.333     256,132,195
24         194.333     256,132,195
48         194.333     256,132,195
72         194.333     256,132,195
96         194.333     256,132,195
120        140.333     128,125,168
144        186.333     256,132,171
153        194.333     256,132,195

s42
iteration  mean_steps  raw_steps
0          194.333     256,132,195
40         194.333     256,132,195
80         194.333     256,132,195
120        154.667     137,132,195
160        187.333     256,132,174
200        194.333     256,132,195
240        154.333     136,132,195
293        178.000     256,83,195
```

Plain read: the quick 256-step rand3 curves do not show an upward survival
trend over each run's own `iteration_0`. s40 and s42 latest checkpoints are
below baseline on this panel; s41 latest is tied with baseline. This is
evaluation plumbing evidence, not learning evidence.

## Frozen s42 Opponent Profile Gate

The trainer-side frozen opponent profile passed against the s42
`iteration_293` checkpoint. This is the gate for short waited
learner-vs-frozen-checkpoint trains.

```text
run_id: curvytron-visual-survival-debug-lz-frozen-s42-iter293-profile
attempt_id: profile-frozen-opponent-s42-iter293-20260510
opponent checkpoint: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s42-wait-livepublish-65536/checkpoints/lightzero/iteration_293.pth.tar
ok: true
called_train_muzero: true
env steps collected: 46
learner train calls: 1
row_count: 91
ego action histogram: {"0": 42, "1": 33, "2": 16}
opponent action histogram: {"0": 80, "1": 11, "2": 0}
mirrored checkpoint: iteration_0.pth.tar
summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-frozen-s42-iter293-profile/attempts/profile-frozen-opponent-s42-iter293-20260510/train/summary.json
```

Plain read: the frozen opponent path reached real `train_muzero`, made one
learner update, collected non-degenerate ego and opponent action telemetry, and
mirrored the first checkpoint. The next active plan is documented in
`docs/working/training/curvytron_snapshot_opponent_interface_2026-05-10.md`:
short waited train with
`--opponent-policy-kind frozen_lightzero_checkpoint`, the s42 `iteration_293`
opponent, frequent checkpoint saves, followed by a compact survival eval curve.

## Frozen s42 Opponent s44/s45 Survival Curves

Current active frozen-opponent train runs were checked directly in the Modal
Volume. Both completed enough to publish checkpoints:

```text
s44 run_id: curvytron-visual-survival-debug-lz-frozen-s42-iter293-s44-4096
s44 attempt_id: train-gpu-l4t4-frozen-s42iter293-4096x32-s44-20260510
s44 visible: iteration_0, iteration_2..iteration_62 every 2, plus iteration_63, ckpt_best
s44 selected eval points: 0,8,16,24,32,40,52,63

s45 run_id: curvytron-visual-survival-debug-lz-frozen-s42-iter293-s45-8192
s45 attempt_id: train-gpu-l4t4-frozen-s42iter293-8192x64-s45-20260510
s45 visible: iteration_0, iteration_4..iteration_104 every 4, ckpt_best
s45 selected eval points: 0,16,32,48,64,80,96,104
```

Train summary telemetry:

```text
s44 ok=true, called_train_muzero=true, problems=[]
  train_elapsed_sec=39.370734
  checkpoint_mirror count=34, copied_now_count=33
  ego_action_histogram={"0":453,"1":245,"2":68}
  opponent_action_histogram={"0":536,"1":230,"2":0}
  done_count=3, collapse_warning=null

s45 ok=true, called_train_muzero=true, problems=[]
  train_elapsed_sec=53.138737
  checkpoint_mirror count=28, copied_now_count=27
  ego_action_histogram={"0":187,"1":203,"2":198}
  opponent_action_histogram={"0":210,"1":304,"2":74}
  done_count=3, collapse_warning=null
```

Eval seed panel was reproducible and varied:

```text
eval_seed_sampler_seed: 20264752
eval_seed_count: 5
eval_seeds: 491165446,1400479014,524684701,1393360233,206817675
max_eval_steps: 1024
source_max_steps: 1024
num_simulations: 8
batch_size: 32
```

Intermediate failed eval attempts:

```text
s44 first eval id: frozen_s42_iter293_s44_survival_curve_seed5_20260510
manifest_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-frozen-s42-iter293-s44-4096/attempts/train-gpu-l4t4-frozen-s42iter293-4096x32-s44-20260510/eval/frozen_s42_iter293_s44_survival_curve_seed5_20260510/manifest_steps1024_seeds1763408531-2037280522-1752135232-245582724-1882988953_20260510T194643Z.json
result: all 40 rows failed before env reset with TypeError:
  _build_visual_survival_configs() missing required opponent_* keyword-only args

s45 first eval id: frozen_s42_iter293_s45_survival_curve_seed5_20260510
result: no remote eval jobs launched; local entrypoint raised:
  ValueError: eval_seeds cannot be combined with eval_seed_count
```

After the eval harness surface changed to provide the missing opponent defaults
and to prefer generated seed panels, the valid strict-load evals completed.
Note: this eval harness currently scores the learner checkpoints in the default
fixed-straight eval env, not against the frozen s42 checkpoint opponent used for
training.

Commands:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id curvytron-visual-survival-debug-lz-frozen-s42-iter293-s44-4096 \
  --attempt-id train-gpu-l4t4-frozen-s42iter293-4096x32-s44-20260510 \
  --eval-id frozen_s42_iter293_s44_survival_curve_seed5_rerun_20260510 \
  --selected-iterations 0,8,16,24,32,40,52,63 \
  --seed 1763408531 \
  --eval-seed-count 5 \
  --eval-seed-rng-seed 20264752 \
  --max-eval-steps 1024 \
  --source-max-steps 1024 \
  --num-simulations 8 \
  --batch-size 32 \
  --parallel \
  --summary-only \
  --quiet-framework-logs

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id curvytron-visual-survival-debug-lz-frozen-s42-iter293-s45-8192 \
  --attempt-id train-gpu-l4t4-frozen-s42iter293-8192x64-s45-20260510 \
  --eval-id frozen_s42_iter293_s45_survival_curve_seed5_20260510 \
  --selected-iterations 0,16,32,48,64,80,96,104 \
  --seed 1763408531 \
  --eval-seed-count 5 \
  --eval-seed-rng-seed 20264752 \
  --max-eval-steps 1024 \
  --source-max-steps 1024 \
  --num-simulations 8 \
  --batch-size 32 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

Manifests:

```text
s44:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-frozen-s42-iter293-s44-4096/attempts/train-gpu-l4t4-frozen-s42iter293-4096x32-s44-20260510/eval/frozen_s42_iter293_s44_survival_curve_seed5_rerun_20260510/manifest_steps1024_seeds491165446-1400479014-524684701-1393360233-206817675_20260510T195005Z.json

s45:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-frozen-s42-iter293-s45-8192/attempts/train-gpu-l4t4-frozen-s42iter293-8192x64-s45-20260510/eval/frozen_s42_iter293_s45_survival_curve_seed5_20260510/manifest_steps1024_seeds491165446-1400479014-524684701-1393360233-206817675_20260510T195125Z.json
```

Survival-step curve, mean over 5 eval starts, cap 1024. Score is secondary
and equals survival reward here.

```text
s44
iteration  ok  mean_steps  median  min  max  mean_score
0          5   145.2       117     48   332  145.2
8          5   150.0       117     72   332  150.0
16         5   144.8       117     46   332  144.8
24         5   138.8       117     77   271  138.8
32         5   166.6       132     97   332  166.6
40         5   145.2       117     48   332  145.2
52         5   145.2       117     48   332  145.2
63         5   145.2       117     48   332  145.2

s45
iteration  ok  mean_steps  median  min  max  mean_score
0          5   145.2       117     48   332  145.2
16         5   145.2       117     48   332  145.2
32         5   135.2       132     97   175  135.2
48         5   154.6       117     95   332  154.6
64         5   142.6       117     38   332  142.6
80         5   160.6       117     97   332  160.6
96         5   125.8       117     97   179  125.8
104        5   133.6       132     97   167  133.6
```

Plain read: neither frozen-opponent train shows a monotonic survival climb over
its own `iteration_0` under the current fixed-straight eval harness. s44 has a
small local bump at `iteration_32`; s45 has local bumps at `iteration_48` and
`iteration_80`, but the latest checkpoint is below baseline. Treat this as
checkpoint/eval plumbing and weak policy signal, not useful CurvyTron learning.
No pytest was run.

## Self-Play Gap Check

Conclusion on 2026-05-10: this trainer is not doing current-policy self-play.

Exact behavior:

- `fixed_straight`: LightZero controls the ego seat. The env always fills the
  opponent with action `1`.
- `frozen_lightzero_checkpoint`: LightZero controls the ego seat. The env fills
  the opponent by running a frozen checkpoint provider.
- The current learner policy is not used for the opponent seat.

Why this is not a small fix: the stock `train_muzero` path calls
`env.step(action)` with one ego action. The env does not receive the live
collector policy object, the current model weights, or the learner update
events. Real current-policy self-play needs a two-seat action contract or a
collector patch that asks the same current policy for both player actions before
stepping the env.

Code observability added: future summaries and env-step telemetry now include
`opponent_training_relation`, `current_policy_self_play: false`, and a plain
`current_policy_self_play_blocker`. This prevents frozen-checkpoint training
from being mistaken for self-play.

## Frozen s42 Opponent Short Runs

Status: the first two waited learner-vs-frozen-checkpoint runs completed.
These are still not full self-play. They answer whether a new ego learner can
improve survival against the frozen s42 `iteration_293` checkpoint.

```text
s44 run_id: curvytron-visual-survival-debug-lz-frozen-s42-iter293-s44-4096
s44 attempt_id: train-gpu-l4t4-frozen-s42iter293-4096x32-s44-20260510
s44 ok: true
s44 checkpoints: iteration_0 through iteration_63, frequent saves
s44 ego actions: {"0": 453, "1": 245, "2": 68}
s44 opponent actions: frozen checkpoint opponent, non-fixed action rows present
s44 summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-frozen-s42-iter293-s44-4096/attempts/train-gpu-l4t4-frozen-s42iter293-4096x32-s44-20260510/train/summary.json

s45 run_id: curvytron-visual-survival-debug-lz-frozen-s42-iter293-s45-8192
s45 attempt_id: train-gpu-l4t4-frozen-s42iter293-8192x64-s45-20260510
s45 ok: true
s45 checkpoints: iteration_0 through iteration_104, frequent saves
s45 ego actions: {"0": 187, "1": 203, "2": 198}
s45 opponent actions: frozen checkpoint opponent, non-fixed action rows present
s45 summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-frozen-s42-iter293-s45-8192/attempts/train-gpu-l4t4-frozen-s42iter293-8192x64-s45-20260510/train/summary.json
```

Active evals:

```text
s44 eval: 0,8,20,30,40,50,60,63 with eight eval starts, cap 1024, 2 searches
s45 eval: 0,12,28,44,60,76,92,104 with eight eval starts, cap 1024, 4 searches
```

Active longer runs:

```text
s46 run_id: curvytron-visual-survival-debug-lz-frozen-s42-iter293-s46-32768
s46 attempt_id: train-gpu-l4t4-frozen-s42iter293-32768x128-s46-20260510
s46 config: 32768 env steps, 128 train iters, 4 collector envs, 4 searches

s47 run_id: curvytron-visual-survival-debug-lz-frozen-s42-iter293-s47-65536
s47 attempt_id: train-gpu-l4t4-frozen-s42iter293-65536x256-s47-20260510
s47 config: 65536 env steps, 256 train iters, 4 collector envs, 2 searches
```

Plain read so far: the frozen-opponent path trains and publishes checkpoints.
We still need the s44/s45 survival curves before claiming any learning signal.
The current self-play gap remains open: these runs train one ego policy against
a frozen checkpoint opponent, not two current policies learning together.

Update: the completed s44/s45 survival curves are recorded above in
`Frozen s42 Opponent s44/s45 Survival Curves`. That completed read supersedes
the `Active evals` note in this section. The survival-first result is weak:
latest s44 matched its `iteration_0` mean under the fixed-straight eval harness,
while latest s45 was below its `iteration_0` mean.

## Frozen s42 Opponent Fixed-Baseline Curves

These evals score learner-vs-frozen-checkpoint training runs in the default
fixed-straight eval env. This is a useful baseline, but it is not the matched
training opponent.

```text
s44 fixed-baseline eval
iteration  mean_steps
0          199.000
8          199.125
20         199.500
30         183.250
40         199.000
50         193.250
60         199.000
63         199.000

s45 fixed-baseline eval
iteration  mean_steps
0          199.000
12         176.125
28         198.000
44         171.000
60         198.250
76         227.500
92         159.875
104        157.750
```

Plain read: s44 is flat. s45 has one middle bump, then ends worse than
`iteration_0`. This is not a clean learning signal.

The eval bug found here was real and fixed: the eval script was still calling
the trainer config builder with its old argument list. It now passes the
opponent config fields, prints aggregate survival tables first, and can run a
matched frozen-checkpoint opponent eval.

Matched-opponent eval smoke passed for s44 `iteration_0` against the frozen
s42 `iteration_293` opponent. Completed matched-opponent curves:

```text
s44 matched frozen-s42 eval
iteration  mean_steps
0          612.000
8          711.375
20         671.500
30         538.125
40         780.500
50         630.500
60         739.625
63         693.125

s45 matched frozen-s42 eval
iteration  mean_steps
0          426.500
12         195.000
28         251.750
44         178.250
60         217.000
76         346.500
92         164.875
104        162.125

s46 matched frozen-s42 eval
iteration  mean_steps
0          428.250
24         166.250
48         166.000
72         415.875
96         560.125
128        165.750
160        469.250
191        297.250

s47 matched frozen-s42 eval
iteration  mean_steps
0          635.125
40         662.625
80         532.875
120        490.250
160        564.875
200        679.500
256        625.750
310        559.375
```

Longer matched-opponent runs completed and were evaluated:

```text
s46 checkpoints: iteration_0 through iteration_191
s47 checkpoints: iteration_0 through iteration_310
```

Fixed-baseline longer-run curves:

```text
s46 fixed-baseline eval
iteration  mean_steps
0          199.000
24         162.125
48         161.750
72         206.250
96         235.875
128        161.125
160        207.125
191        200.500

s47 fixed-baseline eval
iteration  mean_steps
0          199.000
40         167.375
80         198.500
120        182.250
160        198.750
200        199.000
256        198.750
310        178.125
```

Plain fixed-baseline read: s46 has a small middle bump and ends near baseline.
s47 is mostly flat and latest is below baseline.

Plain matched-opponent read: survival is much higher than the fixed-straight
baseline, but unstable. s44 improves, s45 worsens, s46 has a middle bump, and
s47 is high/noisy. This is a signal worth following, not a stable training
proof. It is still learner-vs-frozen-checkpoint, not live current-policy
self-play.

Eval tooling fix after this read:

```text
src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py
```

CurvyTron eval now supports `--compute gpu-l4-t4-cpu40`, matching the Pong
high-CPU eval pattern for broad checkpoint/seed curves. `--summary-only` now
prints only the survival aggregate table, per-seed table, eval seed panel, and
manifest ref. It no longer dumps a large JSON blob after the table.

## Refresh Runs From Best Frozen-Opponent Bumps

These are staged learner-vs-frozen-checkpoint runs. They are not live
current-policy self-play. They were launched to test whether using stronger
frozen opponents gives cleaner survival curves while the real self-play
collector remains open.

```text
s90 run_id: curvytron-visual-survival-debug-lz-refresh-s44iter40-s90-32768
s90 attempt_id: train-gpu-l4t4-refresh-s44iter40-32768x128-s90-20260510
s90 opponent: s44 iteration_40, matched eval peak 780.500 mean steps
s90 function_call_id: fc-01KR9R3V5730SVGR2EYVJ98Y9J
s90 summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-refresh-s44iter40-s90-32768/attempts/train-gpu-l4t4-refresh-s44iter40-32768x128-s90-20260510/train/summary.json

s91 run_id: curvytron-visual-survival-debug-lz-refresh-s46iter96-s91-65536
s91 attempt_id: train-gpu-l4t4-refresh-s46iter96-65536x256-s91-20260510
s91 opponent: s46 iteration_96, matched eval peak 560.125 mean steps
s91 function_call_id: fc-01KR9R3VEHG8XVRA6719BYRK7M
s91 summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-refresh-s46iter96-s91-65536/attempts/train-gpu-l4t4-refresh-s46iter96-65536x256-s91-20260510/train/summary.json

s92 run_id: curvytron-visual-survival-debug-lz-refresh-s47iter200-s92-65536
s92 attempt_id: train-gpu-l4t4-refresh-s47iter200-65536x256-s92-20260510
s92 opponent: s47 iteration_200, matched eval peak 679.500 mean steps
s92 function_call_id: fc-01KR9R3TZYS6S00MCTS76KN3ZS
s92 summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-refresh-s47iter200-s92-65536/attempts/train-gpu-l4t4-refresh-s47iter200-65536x256-s92-20260510/train/summary.json
```

Next eval when checkpoints appear:

```text
Use gpu-l4-t4-cpu40, --parallel, --summary-only, reproducible random eval seed
panels, and report steps survived first.

For each run, evaluate both:
- fixed-straight baseline;
- matched frozen opponent used during training.
```

Launch correction:

```text
The first s90/s91/s92 launches returned "spawned" function call ids but left no
visible summary/checkpoint dirs when polled soon after. Relaunched attached
with --wait-for-train so the local session stays connected and failure/success
is visible.

s90 wait attempt: train-gpu-l4t4-refresh-s44iter40-32768x128-s90-wait-20260510
s90 wait app: ap-qp6LOTNqLH87vW2GFlg06x

s91 wait attempt: train-gpu-l4t4-refresh-s46iter96-65536x256-s91-wait-20260510
s91 wait app: ap-EkCmrjfsXfEuLBy5DATVRP

s92 wait attempt: train-gpu-l4t4-refresh-s47iter200-65536x256-s92-wait-20260510
s92 wait app: ap-X7cYjhPJd7y5zVZUqFhUMx
```

Do not treat the earlier spawned function call ids as proof of running jobs.

Attached run results:

```text
s90 wait result: ok=true, checkpoints through iteration_175, ego actions {"0":274,"1":284,"2":293}, done_count=5.
s91 wait result: ok=true, checkpoints through at least iteration_328, ego actions {"0":720,"1":858,"2":734}, done_count=10.
s92 wait result: ok=true, checkpoints through iteration_434, ego actions {"0":741,"1":620,"2":546}, done_count=15.
```

Eval launches:

```text
s90 fixed eval: refresh_s90_fixed_curve_steps1024_rand8_20260510
s90 matched eval: refresh_s90_matched_s44iter40_curve_steps1024_rand8_20260510
s90 selected iterations: 0,24,48,72,96,128,160,175

s91 fixed eval: refresh_s91_fixed_curve_steps1024_rand8_20260510
s91 matched eval: refresh_s91_matched_s46iter96_curve_steps1024_rand8_20260510
s91 selected iterations: 0,64,128,192,256,320,328

s92 fixed eval: refresh_s92_fixed_curve_steps1024_rand8_20260510
s92 matched eval: refresh_s92_matched_s47iter200_curve_steps1024_rand8_20260510
s92 selected iterations: 0,64,128,192,256,320,384,434
```

Eval seed panel for all six evals:

```text
1093491367,1646752993,983581866,1481646630,264468913,612598383,248211689,31836349
```

Survival-step eval read, mean over 8 eval starts, cap 1024:

```text
s90 fixed-straight baseline
iteration  mean_steps
0          167.000
24         123.250
48         167.750
72         122.250
96         116.125
128        120.500
160        125.500
175        113.875

s90 matched frozen s44 iteration_40 opponent
iteration  mean_steps
0          659.500
24         245.750
48         456.500
72         289.750
96         187.125
128        269.000
160        268.250
175        253.500
```

Plain s90 read: degraded. Fixed-baseline survival falls below the starting
checkpoint, and matched-opponent survival falls hard from the initial
checkpoint.

```text
s91 fixed-straight baseline
iteration  mean_steps
0          132.125
64         108.125
128        132.250
192        162.625
256        125.000
320        169.750
328        167.875

s91 matched frozen s46 iteration_96 opponent
iteration  mean_steps
0          181.000
64         314.750
128        333.375
192        365.250
256        232.125
320        213.625
328        464.750
```

Plain s91 read: fixed-baseline survival is mostly flat/noisy. Matched-opponent
survival has useful but noisy improvement, with the latest checkpoint best on
this panel.

```text
s92 fixed-straight baseline
iteration  mean_steps
0          132.125
64         167.000
128        114.750
192        158.875
256        167.000
320        167.000
384        140.375
434        162.000

s92 matched frozen s47 iteration_200 opponent
iteration  mean_steps
0          503.125
64         358.750
128        331.375
192        491.125
256        565.500
320        579.375
384        589.000
434        541.750
```

Plain s92 read: fixed-baseline survival is flat. Matched-opponent survival is
the best staged-refresh signal so far, especially `iteration_256` through
`iteration_384`, but the claim is narrow because the opponent is frozen.

Overall plain read: these are learner-vs-frozen-checkpoint runs, not true
self-play. Survival steps are the main metric. The next move should be either a
staged refresh from s92 `iteration_384` or implementation of true
two-seat/current-policy self-play, preferably both in parallel.

## Worker D Refresh Poll - 2026-05-10 16:10 EDT

Follow-up poll for the staged refresh runs. These are learner-vs-frozen
checkpoint runs, not self-play.

```text
s90 run_id: curvytron-visual-survival-debug-lz-refresh-s44iter40-s90-32768
s90 attempt_id: train-gpu-l4t4-refresh-s44iter40-32768x128-s90-20260510
s90 frozen opponent: curvytron-visual-survival-debug-lz-frozen-s42-iter293-s44-4096/checkpoints/lightzero/iteration_40.pth.tar
s90 call_id: fc-01KR9R3V5730SVGR2EYVJ98Y9J
s90 poll: no train summary dir and no stable checkpoint root visible yet

s91 run_id: curvytron-visual-survival-debug-lz-refresh-s46iter96-s91-65536
s91 attempt_id: train-gpu-l4t4-refresh-s46iter96-65536x256-s91-20260510
s91 frozen opponent: curvytron-visual-survival-debug-lz-frozen-s42-iter293-s46-32768/checkpoints/lightzero/iteration_96.pth.tar
s91 call_id: fc-01KR9R3VEHG8XVRA6719BYRK7M
s91 poll: no train summary dir and no stable checkpoint root visible yet

s92 run_id: curvytron-visual-survival-debug-lz-refresh-s47iter200-s92-65536
s92 attempt_id: train-gpu-l4t4-refresh-s47iter200-65536x256-s92-20260510
s92 frozen opponent: curvytron-visual-survival-debug-lz-frozen-s42-iter293-s47-65536/checkpoints/lightzero/iteration_200.pth.tar
s92 call_id: fc-01KR9R3TZYS6S00MCTS76KN3ZS
s92 poll: no train summary dir and no stable checkpoint root visible yet
```

No eval launched from this poll because none of the three refresh runs had a
visible checkpoint directory. Next action: poll the three `summary_ref` paths
and stable checkpoint roots again. When checkpoints appear, launch two eval
curves per run with `gpu-l4-t4-cpu40`, `--parallel`, and `--summary-only`:
fixed-straight baseline and matched frozen-opponent eval using the training
opponent checkpoint.

## 2026-05-10 Late CurvyTron Read

Short version: the fixed-opponent path is still not learning in a useful way.
The main next path is true current-policy two-seat training, not another long
fixed-opponent run.

Two-seat status:

- local current-policy two-seat smoke passed with the tiny numpy learner:
  `src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py`;
- LightZero-facing two-seat smoke exists:
  `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`;
- the LightZero-facing smoke still needs a Modal runtime run with installed
  LightZero before it can be claimed as a LightZero policy/learner smoke.

s93 was an old anonymous-visual staged refresh run from s92 `iteration_384`.
It completed, but its eval did not rescue the lane.

```text
s93 run_id: curvytron-visual-survival-debug-lz-refresh-s92iter384-s93-131072
s93 attempt_id: train-gpu-l4t4-refresh-s92iter384-131072x512-s93-wait-20260510
s93 training opponent: frozen s92 iteration_384
s93 checkpoints: through iteration_584
s93 action histogram: ego {"0":810,"1":1049,"2":3001}
s93 current_policy_self_play: false
```

Mean steps survived, cap 1024:

```text
s93 fixed-straight eval, 16 starts
iteration_0   143.688
iteration_128 132.375
iteration_256 148.500
iteration_384 134.812
iteration_512 148.375
iteration_584 131.375

s93 matched frozen-opponent eval, 16 starts
iteration_0   306.750
iteration_128 309.375
iteration_256 204.875
iteration_384 250.688
iteration_512 284.188
iteration_584 173.625
```

Plain read: s93 is flat against fixed-straight and drifts down against the
matched frozen opponent. It is not the path forward.

s100 was the first player-aware fixed-straight run. The observation now marks
the controlled player separately from the other player. That fixed an important
input bug, but the fixed-opponent training curve is still flat.

```text
s100 run_id: curvytron-visual-survival-player-aware-fixed-s100-131072
s100 attempt_id: train-gpu-l4t4-player-aware-fixed-131072x512-s100-wait-20260510
s100 opponent: fixed straight
s100 checkpoints: through iteration_520
s100 observation_schema_id: curvyzero_stacked_debug_occupancy_gray64_player_aware_survival_time/v1
s100 action histogram: ego {"0":893,"1":918,"2":752}; opponent always action 1
s100 current_policy_self_play: false
```

Mean steps survived against fixed-straight, 64 random starts, cap 1024:

```text
iteration_0   178.531
iteration_128 164.625
iteration_256 170.172
iteration_384 160.875
iteration_512 160.375
iteration_520 171.906
```

Plain read: player-aware input alone did not make fixed-opponent CurvyTron
learn. Keep the player-aware observation as the correct input, but stop treating
fixed-straight training as the main route.

New runs launched for more artifacts while self-play integration proceeds:

```text
s101: curvytron-visual-survival-player-aware-fixed-s101-262144
attempt: train-gpu-l4t4-player-aware-fixed-262144x1024-s101-wait-20260510
purpose: longer player-aware fixed-opponent control

s102: curvytron-visual-survival-player-aware-fixed-s102-sim8-131072
attempt: train-gpu-l4t4-player-aware-fixed-131072x512-s102-sim8-wait-20260510
purpose: player-aware fixed-opponent control with more search per move
```

Do not claim these are self-play. They are controls while the actual two-seat
current-policy path is built.

Tooling note: tried a `gpu-l4-t4-cpu64` eval function, but Modal rejected it:
one function can request at most 40 CPUs. Keep CurvyTron eval on
`gpu-l4-t4-cpu40` and scale by launching many independent eval workers.

s101 and s102 completed cleanly and were sent to fixed-straight eval:

```text
s101: ok=true, current_policy_self_play=false, checkpoints through iteration_1071,
ego actions {"0":1690,"1":1762,"2":1367}, done_count=26.
eval: s101_fixed64, iterations 0,256,512,768,1024,1071, 64 starts.

s102: ok=true, current_policy_self_play=false, checkpoints through iteration_540,
ego actions {"0":1398,"1":919,"2":599}, done_count=16.
eval: s102_fixed64, iterations 0,128,256,384,512,540, 64 starts.
```

Artifact/status check from the eval worker at 20:45 EDT:

```text
s93 fixed16 manifest:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-refresh-s92iter384-s93-131072/attempts/train-gpu-l4t4-refresh-s92iter384-131072x512-s93-wait-20260510/eval/s93_fixed16

s93 matched16 manifest:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-refresh-s92iter384-s93-131072/attempts/train-gpu-l4t4-refresh-s92iter384-131072x512-s93-wait-20260510/eval/s93_matched16

s100 fixed64 manifest:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-player-aware-fixed-s100-131072/attempts/train-gpu-l4t4-player-aware-fixed-131072x512-s100-wait-20260510/eval/s100_fixed64/manifest_steps1024_seedsn64_e59c2859a585_20260510T203258Z.json

s101 fixed64 manifest:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-player-aware-fixed-s101-262144/attempts/train-gpu-l4t4-player-aware-fixed-262144x1024-s101-wait-20260510/eval/s101_fixed64/manifest_steps1024_seedsn64_e59c2859a585_20260510T204124Z.json

s102 fixed64 manifest:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-player-aware-fixed-s102-sim8-131072/attempts/train-gpu-l4t4-player-aware-fixed-131072x512-s102-sim8-wait-20260510/eval/s102_fixed64/manifest_steps1024_seedsn64_e59c2859a585_20260510T204136Z.json
```

Mean steps survived against fixed-straight, 64 random starts, cap 1024:

```text
s101 fixed64
iteration_0    154.812
iteration_256  158.391
iteration_512  171.141
iteration_768  166.047
iteration_1024 170.141
iteration_1071 171.375

s102 fixed64
iteration_0   178.531
iteration_128 166.984
iteration_256 177.219
iteration_384 178.625
iteration_512 177.422
iteration_540 174.719
```

Plain read: s101 shows only a small late lift. s102 is flat. Both are still
fixed-opponent controls, not current-policy self-play. The compact status table
lives in
`docs/working/coach_north_star_2026-05-10.md`.

## Two-Seat Current-Policy Smoke

The fixed-opponent runs above remain controls. The active two-seat integration
path now has a bounded installed-LightZero smoke:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
  --seed 0 \
  --batch-size 1 \
  --steps 4 \
  --num-simulations 2 \
  --learner-updates 1 \
  --output summary
```

Result:

```text
Modal app: ap-jGJMCz977NA7xgWmhG62Vu
ok: true
lightzero_policy_status: ok
steps_survived: 4
problems: []
replay.row_count: 8
replay.sample.players: [0, 1]
replay.sample.observation_batch_shape: [8, 4, 64, 64]
replay.sample.next_observation_batch_shape: [8, 4, 64, 64]
replay.sample.policy_batch_shape: [8, 3]
replay.sample.reward_sum: 8.0
learner_forward.status: run
learner_forward.ok: true
learner_forward.api: MuZeroPolicy.learn_mode.forward
learner_forward.blocker: null
```

Plain read: this proves one current installed `MuZeroPolicy` object can choose
both CurvyTron seats before `VectorMultiplayerEnv.step(joint_action[B,P])`,
record both-player replay rows, sample them, and call `learn_mode.forward` in
the bounded smoke. It is still not `train_muzero`, not LightZero collector
self-play, and not a distributed actor weight-refresh implementation.

### Real Optimizer-Step Two-Seat Smoke

The two-seat smoke now has a tiny real train mode while keeping the old no-op
smoke available.

Compile:

```bash
uv run python -m py_compile \
  src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py \
  src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py
```

Result: pass.

Command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
  --seed 0 \
  --batch-size 1 \
  --steps 8 \
  --num-simulations 2 \
  --learner-updates 1 \
  --allow-optimizer-step \
  --run-id curvytron-two-seat-realtrain-smoke-s0-20260510 \
  --attempt-id realtrain-steps8-updates1-20260510 \
  --output summary
```

Result:

```text
Modal app: ap-dohm6MpfKlDoZkU3LH3Ixx
ok: true
mode: bounded_two_seat_lightzero_collect_replay_real_train_smoke
steps_survived: 8
replay.row_count: 16
replay.sample.players: [0, 1]
learner_forward.status: updated
learner_forward.optimizer_step: allowed
learner_forward.model_hash_before: d944a3b73e3e14af
learner_forward.model_hash_after: ffc34a228971d6b1
learner_forward.model_parameters_changed: true
```

Checkpoint refs:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/iteration_0.pth.tar
training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/iteration_1.pth.tar
training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/latest.pth.tar
```

Next command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --run-id curvytron-two-seat-realtrain-smoke-s0-20260510 \
  --attempt-id realtrain-steps8-updates1-20260510 \
  --checkpoint-refs training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/iteration_0.pth.tar,training/lightzero-curvytron-visual-survival/curvytron-two-seat-realtrain-smoke-s0-20260510/checkpoints/lightzero/latest.pth.tar \
  --output summary
```

Plain read: weights really changed in the bounded two-seat smoke. This is still
not a full `train_muzero` self-play trainer.

Strict-load compatibility note, 2026-05-10: old two-seat smoke checkpoints
written before the visual model config was aligned may not strict-load in the
CurvyTron eval harness. New two-seat smoke checkpoints should use the eval
contract: `image_channel=4`, `frame_stack_num=1`, and observation shape
`[4,64,64]`, with the same self-supervised MuZero model heads enabled.

### Batchfix Two-Seat Smoke Status

Latest facts:

- Fixed-opponent CurvyTron controls remain flat or weak. They are control runs
  only, not self-play.
- The two-seat current-policy Modal smoke now passes with real optimizer
  updates after the learner batch next-observation fix.
- Batchfix checkpoints strict-load in eval.
- Eval returned the same `176.5` mean steps for `iteration_0` and
  `iteration_2`.

Plain read: this proves the two-seat smoke pipeline can collect, batch, update,
checkpoint, strict-load, and eval. It does not prove learning.

Resolved blocker: batch-size scaling no longer crashes on ended vector rows.
After each env step, the trainer records the terminal transition, autoresets
done rows, and refreshes only those visual-stack rows before the next step.

### Two-Seat Batch Scaling And Eval Curves

Batch 8 smoke:

```text
run_id: curvytron-two-seat-realupdate-batch8-s4-64x8
attempt_id: two-seat-batch8-steps64-updates8-20260510
ok: true
collection steps: 64
replay rows: 1024
checkpoints: iteration_0..iteration_8
```

Eval, 16 random starts, 512-step cap:

| checkpoint | mean steps survived |
| --- | ---: |
| iteration_0 | 184.0 |
| iteration_1 | 166.125 |
| iteration_2 | 184.0625 |
| iteration_4 | 172.75 |
| iteration_6 | 172.3125 |
| iteration_8 | 183.5625 |

Batch 16 / sim2 smoke:

```text
run_id: curvytron-two-seat-realupdate-batch16-s5-256x32-sim2
attempt_id: two-seat-batch16-steps256-updates32-sim2-20260510
ok: true
collection steps: 256
replay rows: 8192
checkpoints: iteration_0..iteration_32
```

Eval, 16 random starts, 512-step cap:

| checkpoint | mean steps survived |
| --- | ---: |
| iteration_0 | 187.5 |
| iteration_1 | 185.062 |
| iteration_2 | 185.75 |
| iteration_4 | 185.25 |
| iteration_8 | 185.062 |
| iteration_16 | 185.562 |
| iteration_24 | 185.0 |
| iteration_32 | 185.25 |

Batch 16 / sim4 smoke:

```text
run_id: curvytron-two-seat-realupdate-batch16-s6-256x32-sim4
attempt_id: two-seat-batch16-steps256-updates32-sim4-20260510
ok: true
collection steps: 256
replay rows: 8192
checkpoints: iteration_0..iteration_32
```

Eval, 16 random starts, 512-step cap:

| checkpoint | mean steps survived |
| --- | ---: |
| iteration_0 | 147.625 |
| iteration_1 | 149.562 |
| iteration_2 | 149.062 |
| iteration_4 | 153.75 |
| iteration_8 | 153.688 |
| iteration_16 | 149.875 |
| iteration_24 | 151.375 |
| iteration_32 | 148.688 |

Plain read: the two-seat pipeline now collects, updates, checkpoints, and evals
at larger batch sizes, but these curves are flat. The most likely next issue is
that this bounded trainer still collects one replay batch and then does many
updates on that same fixed data. The next trainer step should repeat
collect -> update -> checkpoint so later data comes from the updated current
policy.

### Iterative Current-Policy Two-Seat Smoke

Implemented the smallest local refresh loop in
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py` and its
Modal wrapper:

```text
repeat outer_iterations times:
  collect current-policy actions for both seats with the same live policy object
  build/sample replay from that iteration's rows
  run updates_per_iteration learner updates on the same policy object
  checkpoint iteration_N after the learner phase
```

The original one-shot smoke remains the default with `--outer-iterations 1`.
New knobs are `--outer-iterations`, `--collect-steps-per-iteration`, and
`--updates-per-iteration`; `--collect-steps-per-iteration` defaults to `--steps`
and `--updates-per-iteration` defaults to `--learner-updates`.

Small iterative smoke command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
  --seed 0 \
  --batch-size 1 \
  --outer-iterations 2 \
  --collect-steps-per-iteration 4 \
  --updates-per-iteration 1 \
  --num-simulations 2 \
  --allow-optimizer-step \
  --run-id curvytron-two-seat-iterative-smoke-s0-20260510 \
  --attempt-id iterative-2x4x1-20260510 \
  --output summary
```

Expected checkpoint shape: `iteration_0.pth.tar` before learning, then
`iteration_1.pth.tar`, `iteration_2.pth.tar`, `latest.pth.tar`, and
`ckpt_best.pth.tar` after the two outer iterations. The summary now includes
per-iteration survival reward, replay row/sample shape, action counts by seat,
learner update status, and checkpoint refs.

Verification run for this code change:

```bash
python3 -m py_compile \
  src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py \
  src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py
```

Result: pass.

Plain read: the fixed-data replay issue is addressed for this bounded local
smoke. This still does not make the run full LightZero current-policy
self-play: it bypasses `train_muzero` and LightZero's collector, and it has not
yet shown non-flat eval curves.
