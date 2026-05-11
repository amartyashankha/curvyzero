# CurvyTron Visual Survival Training Worker Handoff - 2026-05-10

Purpose: answer whether CurvyTron visual survival is ready to move from bounded
profile work toward a real LightZero trainer run.

## Verdict

The bounded installed-LightZero visual profile gate now passes.

This is still not a trainer run and not a learning claim. It proves only that
the current debug-fidelity visual survival surface can exercise:

```text
wrapper-stacked debug visual collect
-> MuZeroPolicy.eval_mode.forward MCTS/search
-> replay row build
-> replay sample/batch
-> MuZeroPolicy.learn_mode.forward loss
```

The safe real-trainer path is to add a separate, clearly labeled trainer
attempt artifact owned by the training worker. Do not put the trainer call into
the profile file.

Latest Coach-lane status:

- Fixed-opponent CurvyTron runs are flat or weak controls. They are not
  self-play.
- The iterative two-seat run
  `curvytron-two-seat-iterative-b8-s7-4x32-u2-sim2` passed mechanically with 4
  collect/update rounds, 2048 replay rows, and 5 checkpoints, but 32-start eval
  means fell from `181.28125` at `iteration_0` to `170.71875` by
  `iteration_4`.
- Do not scale more CurvyTron runs until two blockers are fixed: `target_value`
  must become discounted survival return rather than immediate reward, and the
  learner batch-size path must stop inheriting the profile's hard
  `policy.batch_size=2` / `_learn_mode_batches` slicing behavior.

## Self-Play Status

The current CurvyTron debug visual survival lane is not current-policy
self-play. LightZero controls one ego seat, and the wrapper fills the opponent
seat with the fixed straight opponent policy. Weak improvement may therefore be
partly explained by learning against a narrow fixed opponent rather than a
moving current-policy opponent.

Simplest correct next implementation step: keep the single-ego LightZero
wrapper shape, but add a versioned policy-backed opponent that can load the
latest/current checkpoint snapshot and choose the non-ego action. Do not start
with full joint-action MCTS or a custom all-player LightZero collector.

## What Passed

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_profile \
  --seed 0 \
  --steps 4 \
  --num-simulations 2
```

Modal app:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-EubQiSXUN6L5PD1q0wfkfC
```

Installed runtime:

```text
LightZero: 0.2.0
DI-engine: 0.5.3
torch: 2.11.0
gym: 0.25.1
numpy: 1.26.4
```

Profile surface:

```text
env.type: curvyzero_stacked_debug_visual_survival_lightzero
observation_shape: [4,64,64]
model_type: conv
action_space_size: 3
collector_env_num: 1
num_simulations: 2
batch_size: 2
num_unroll_steps: 1
reward_schema_id: curvyzero_survival_time/v0
source_fidelity_claim: none
debug_fidelity_only: true
called_train_muzero: false
trainer_claim: none
quality_claim: none
```

Passed stages:

```text
collect rows: 4
MCTS/search API: MuZeroPolicy.eval_mode.forward
MCTS ok_count: 4
replay row_count: 4
sample observation_batch_shape: [2,4,64,64]
sample action_batch_shape: [2,1]
sample target_reward_shape: [2,1]
sample target_value_shape: [2,2]
sample target_policy_shape: [2,2,3]
learner API: MuZeroPolicy.learn_mode.forward
learner status: run
optimizer_step: blocked_by_noop_patch
model_parameters_changed: false
model_state_restored: true
```

## Remaining Trainer Blocker

Superseded: the separate trainer artifact now exists and successful train runs
have been launched. Keep this section only as historical context for the
pre-trainer profile gate.

Exact missing call:

```text
lzero.entry.train_muzero
```

Recommended owner file:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train_attempt.py
```

Do not add this call to:

```text
src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py
```

That profile file should remain a bounded no-train gate.

## Training Worker Path

Use the passing profile as the prerequisite gate, then add a separate trainer
attempt wrapper with the same surface:

```text
env.type: curvyzero_stacked_debug_visual_survival_lightzero
env.import_names:
  ["curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env"]
model_type: conv
observation_shape: [4,64,64]
action_space_size: 3
frame_stack_num: 1
frame_stack_owner: curvyzero_wrapper_local_debug_frame_stack
collector_env_num: 1 for the first trainer attempt
evaluator_env_num: 1 for the first trainer attempt
num_simulations: low first, then scale
reward_schema_id: curvyzero_survival_time/v0
terminal outcome bonus: 0.0
loser penalty: 0.0
winner bonus: 0.0
source_fidelity_claim: none
debug_fidelity_only: true
```

The trainer wrapper should call `lzero.entry.train_muzero` only in that trainer
artifact. It should write outputs to a labeled debug visual survival experiment
directory and report itself as a trainer plumbing attempt until a separate
checkpoint/eval or baseline story exists.

## Must Not Claim

- Do not claim full CurvyTron training from the profile.
- Do not claim source-fidelity visuals.
- Do not add terminal outcome shaping, winner bonus, or loser penalty.
- Do not claim policy quality or learning from a no-train profile.
- Do not reuse the scalar/ray survival trainer scaffold as the visual trainer.
- Do not report a long visual survival run until the separate trainer artifact
  calls `lzero.entry.train_muzero` and records its own result.

## Urgent Training Risk Register

Most likely CurvyTron visual survival training issues to check first:

1. Action collapse: ego action histogram may collapse to one action even when
   `train_muzero` succeeds.
2. Checkpoint pressure: frequent checkpoints can create many ~96 MB files; long
   Hail Mary runs should use a non-1 cadence unless storage is confirmed safe.
3. Reward loophole: survival-only reward can favor passive behavior against the
   fixed straight opponent.
4. Input contract drift: model/env must stay on `float32[4,64,64]`, `image_channel=4`,
   `frame_stack_num=1`, wrapper-owned stack.
5. Source horizon mismatch: `source_max_steps` may truncate episodes before a
   useful survival signal appears.
6. Search budget mismatch: low `num_simulations` may under-search; high values
   may spend credits without improving the first signal.
7. Batch-size mismatch: small batches can be noisy; larger batches can slow the
   learner and change replay pressure.
8. Env horizon vs training bound: installed LightZero did not strictly stop at
   the requested `max_train_iter`; treat `max_env_step` as the effective bound.
9. Eval gap: current train artifacts are not quality claims until checkpoint
   eval/summary paths compare against same-run baseline checkpoints.
10. Summary path drift: every launch must record `summary_ref`,
    `action_observability_ref`, checkpoint root, app id, and call id before any
    claim is made.

## Eval Harness Update

The minimal CurvyTron checkpoint eval harness now exists:

```text
src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py
```

It loads mirrored LightZero MuZero checkpoints, strict-loads the policy model,
runs the registered stacked debug visual survival env, and writes per-checkpoint
JSON plus a manifest table under the attempt eval root.

Curve command shape:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id <run_id> \
  --attempt-id <attempt_id> \
  --selected-iterations 0,100,200 \
  --seed <seed> \
  --max-eval-steps 256 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

Small s0 smoke eval was launched and completed:

```text
run_id: curvytron-visual-survival-debug-lz-s0
attempt_id: train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510
checkpoint: iteration_0
seed: 0
cap: 4
strict_load: true
steps_survived: 4
total_reward: 4.0
action_histogram: {"0": 4}
terminal_reason: cap
manifest_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/attempts/train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510/eval/checkpoint_curve/manifest_steps4_seeds0_20260510T185925Z.json
```

No pytest was run for this lane.

## Live Checkpoint Visibility Update

The trainer now publishes checkpoints during the run, not only after the run
ends. The fix lives in:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Implementation: patch LightZero `BaseLearner.save_checkpoint` during train
runs; after each checkpoint save, rescan `lightzero_exp`, mirror checkpoints
into the stable run checkpoint root, write `live_checkpoint_publish.json`, and
call `runs_volume.commit()`.

Stable eval-visible checkpoint root:

```text
training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero/
```

Scope reminder: this trainer is learned ego against a fixed straight opponent.
It is not multiplayer self-play.

Smoke launch:

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
checkpoint_mirror count: 45
latest visible iteration: iteration_43.pth.tar
Volume proof command:
  modal volume ls curvyzero-runs training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s16-livepublish-smoke/checkpoints/lightzero
Volume proof included:
  iteration_0.pth.tar
  iteration_43.pth.tar
```

Small Hail Mary eval curve launched and completed:

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
iteration_0 seed 16: strict_load=true, steps_survived=173, terminal_reason=survivor_win
iteration_43 seed 16: strict_load=true, steps_survived=173, terminal_reason=survivor_win
iteration_0 seed 17: strict_load=true, steps_survived=164, terminal_reason=survivor_win
iteration_43 seed 17: strict_load=true, steps_survived=180, terminal_reason=survivor_win
```

## Urgent Launches

Existing current launches from the wider wave, not duplicated here:

```text
s10/s11: 32768x128
s12: 65536x256
```

Two additional background runs were launched under the proven trainer module:

```text
variant run:
  run_id: curvytron-visual-survival-debug-lz-s14-sim16-32k
  attempt_id: train-gpu-l4t4-survival-debug-32768x128-s14-sim16-20260510
  Modal app: ap-nYLT88ZGlSxGNxo8CywJKA
  function_call_id: fc-01KR95MRWF547QDNDZAC4XWMYG
  seed: 14
  max_env_step: 32768
  max_train_iter: 128
  source_max_steps: 1024
  num_simulations: 16
  batch_size: 32
  save_ckpt_after_iter: 4
  summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s14-sim16-32k/attempts/train-gpu-l4t4-survival-debug-32768x128-s14-sim16-20260510/train/summary.json
  action_observability_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s14-sim16-32k/attempts/train-gpu-l4t4-survival-debug-32768x128-s14-sim16-20260510/train/action_observability.json
```

```text
Hail Mary long run:
  run_id: curvytron-visual-survival-debug-lz-s13-hailmary131k
  attempt_id: train-gpu-l4t4-survival-debug-131072x512-s13-hailmary-20260510
  Modal app: ap-grD5fS18CXOzi5g0d6KrdA
  function_call_id: fc-01KR95MS06C0W8HNHMRSSZYRZY
  seed: 13
  max_env_step: 131072
  max_train_iter: 512
  source_max_steps: 2048
  num_simulations: 8
  batch_size: 32
  save_ckpt_after_iter: 8
  summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s13-hailmary131k/attempts/train-gpu-l4t4-survival-debug-131072x512-s13-hailmary-20260510/train/summary.json
  action_observability_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s13-hailmary131k/attempts/train-gpu-l4t4-survival-debug-131072x512-s13-hailmary-20260510/train/action_observability.json
```

Both launches are still debug-fidelity only, survival-reward only, and no
policy-quality claim. First gate after completion: fetch `summary.json` and
`action_observability.json`, confirm `ok`, `called_train_muzero`,
checkpoint count/root, `[4,64,64]` surface, and non-collapsed ego action
histogram before scheduling eval.
