# CurvyTron LightZero Visual Profile Blocker Brief - 2026-05-10

Purpose: keep the next CurvyTron/LightZero step focused on visual plumbing, not
scalar survival config. This is not a trainer run and not a learning claim.

## Exact Blocker

The scalar survival adapter is not enough for the CurvyTron visual path.

Current visual shape facts:

```text
existing debug visual env: float32[1,64,64]
visual MuZero target: float32[4,64,64]
proof that LightZero stacks CurvyTron debug visual frames: none
```

Small safe patch added a clearly labeled wrapper-stacked debug visual survival
adapter:

```text
env.type: curvyzero_stacked_debug_visual_survival_lightzero
import_names: ["curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env"]
observation: float32[4,64,64]
raw frame source: curvyzero_debug_occupancy_gray64/v0, float32[1,64,64]
frame_stack_owner: curvyzero_wrapper_local_debug_frame_stack
reward_schema_id: curvyzero_survival_time/v0
truth level: debug-fidelity only
```

This proves shape plumbing only. It does not prove LightZero frame stacking,
training, or policy quality.

## Current Safe Command

Run only the shape/collect/replay-sample smoke:

```text
uv run python -m curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke
```

Expected status:

```text
ok: true
mode: no_train_stacked_debug_visual_survival_collect_replay_sample_only
observation_shape: [4,64,64]
reward_schema_id: curvyzero_survival_time/v0
mcts_search: not_run
learner_profile: not_run
called_train_muzero: false
```

## Current Profile Artifact

The bounded debug-fidelity visual profile now exists:

```text
CurvyTron visual profile rows
-> LightZero MuZero MCTS/search
-> visual replay row
-> replay sample/batch
-> no-step MuZeroPolicy.learn_mode.forward loss profile
```

Required constraints:

```text
env.type: curvyzero_stacked_debug_visual_survival_lightzero
observation_shape: [4,64,64]
model_type: conv
num_simulations: 2
collector_env_num: 1
reward_schema_id: curvyzero_survival_time/v0
terminal outcome bonus: 0.0
loser penalty: 0.0
source_fidelity_claim: none
debug_fidelity_only: true
no train_muzero call
no optimizer step
```

## Current Status

Implemented command:

```text
uv run python -m curvyzero.training.curvyzero_stacked_debug_visual_survival_profile \
  --seed 0 \
  --steps 4 \
  --num-simulations 2 \
  --require-installed-lightzero
```

Installed Modal command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_profile \
  --seed 0 \
  --steps 4 \
  --num-simulations 2
```

Installed runtime result:

```text
app: ap-EubQiSXUN6L5PD1q0wfkfC
LightZero: 0.2.0
DI-engine: 0.5.3
torch: 2.11.0
gym: 0.25.1
numpy: 1.26.4
profile rows: 4
MCTS/search: ran through MuZeroPolicy.eval_mode.forward
observation to policy: [1,4,64,64]
replay row: built
replay sample/batch: built, observation_batch [4,4,64,64]
learner sample/batch: built, observation_batch [2,4,64,64]
learner forward/loss: ran through MuZeroPolicy.learn_mode.forward
optimizer step: blocked by no-op patch
scheduler step: blocked by no-op patch
target update: blocked by no-op patch
model_parameters_changed: false
model_state_restored: true
called_train_muzero: false
trainer_claim: none
quality_claim: none
ok: true
elapsed: 7.283221s
LightZero setup: 7.178879s
policy_search: 0.036811s for 4 eval/search calls
env_step_render_stack: 0.001278s
replay_row_build: 0.000184s
replay_sample_batch: 0.000077s
learner_forward_loss: 0.067410s
```

What this proves:

```text
installed LightZero conv MuZeroPolicy can accept CurvyTron debug visual
float32[4,64,64] observations for eval-mode MCTS/search; the profile can turn
that result into a visual replay row and sampled batch; installed
MuZeroPolicy.learn_mode.forward can compute a loss on the sampled [B,4,64,64]
batch when optimizer/scheduler/target updates are patched to no-op and the
model state is restored afterward.
```

Historical profile-only caveat before the trainer update:

```text
No train_muzero call.
No long trainer config/run artifact.
No real optimizer update.
No optimizer step.
No checkpoint/eval.
No source-fidelity visual tensor.
No policy-quality or training claim.
```

Historical blocker before the trainer update:

```text
missing call: lzero.entry.train_muzero
missing artifact: separate trainer config/run wrapper
must not live in: src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py
recommended owner file:
  src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train_attempt.py
```

## Update: Trainer Run Completed

The separate trainer artifact now exists and has called real installed
LightZero `train_muzero` on the stacked debug visual survival env.

```text
trainer module:
  src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
env action telemetry:
  src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_smoke.py
```

First waited train attempt exposed the stack blocker:

```text
attempt_id: train-gpu-l4t4-survival-debug-4096x32-wait-20260510
called_train_muzero: true
status: failed
problem: expected input[1,16,64,64] to have 4 channels
fix: policy.model.frame_stack_num=1 and policy.model.image_channel=4
```

Successful train command:

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
summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/attempts/train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510/train/summary.json
action rows: 461
ego_action_histogram: {"0": 226, "1": 108, "2": 127}
collapse_warning: null
mirrored checkpoints: 75
checkpoint root: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/checkpoints/lightzero/
```

This removes the "no trainer call" blocker for debug-fidelity visual survival
plumbing. Remaining blockers are now evaluation and quality gates: no
source-fidelity visual tensor, no independent scorecard/eval, and no claim that
the learned policy is good.

Full run note:

```text
docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md
```
