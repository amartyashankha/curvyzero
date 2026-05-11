# CurvyTron Native Train MuZero Fresh Run Check - 2026-05-10

Decision: skipped a fresh launch.

Reason: existing same-day native `train_muzero` fixed-straight runs are already
longer than smoke, completed successfully, and published frequent checkpoints.
Launching another run in the same lane would be redundant until those
checkpoints are evaluated.

## Requested Lane

Use the native Modal wrapper:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Keep the Pong-like contract:

```text
visual survival env
env-owned fixed-straight opponent
native LightZero collector/replay/MCTS/learner/checkpointing
no custom two-seat trainer
```

The existing runs below satisfy that contract:

```text
trainer_entrypoint: lzero.entry.train_muzero
env_type: curvyzero_stacked_debug_visual_survival_lightzero
env_id: CurvyZeroStackedDebugVisualSurvivalLightZero-v0
opponent_policy_kind: fixed_straight
opponent_training_relation: learner_vs_fixed_straight
reward_schema_id: curvyzero_survival_time/v0
observation_shape: [4, 64, 64]
debug_fidelity_only: true
current_policy_self_play: false
```

## Existing Native Runs

Best available completed fixed-straight run:

```text
run_id: curvytron-visual-survival-player-aware-fixed-s101-262144
attempt_id: train-gpu-l4t4-player-aware-fixed-262144x1024-s101-wait-20260510
status: completed
started_at: 2026-05-10T20:32:32.160089Z
ended_at: 2026-05-10T20:36:14.210965Z
modal_task_id: ta-01KR9SEJCFPZ1D6MEWYYAGVTFP
compute: gpu-l4-t4
seed: 101
max_env_step: 262144
max_train_iter: 1024
source_max_steps: 1024
num_simulations: 4
batch_size: 32
save_ckpt_after_iter: 16
summary ok: true
train_result ok: true
return_type: MuZeroPolicy
trainer_entrypoint: lzero.entry.train_muzero
ego_action_histogram: {"0": 1690, "1": 1762, "2": 1367}
final_rewards: [125.0, 125.0, 125.0, 125.0]
latest visible checkpoint: iteration_1071.pth.tar
checkpoint root: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-player-aware-fixed-s101-262144/checkpoints/lightzero
summary_ref: training/lightzero-curvytron-visual-survival/curvytron-visual-survival-player-aware-fixed-s101-262144/attempts/train-gpu-l4t4-player-aware-fixed-262144x1024-s101-wait-20260510/train/summary.json
```

Additional completed fixed-straight native runs:

```text
run_id: curvytron-visual-survival-player-aware-fixed-s100-131072
attempt_id: train-gpu-l4t4-player-aware-fixed-131072x512-s100-wait-20260510
status: completed
started_at: 2026-05-10T20:28:35.936819Z
ended_at: 2026-05-10T20:30:57.854396Z
modal_task_id: ta-01KR9S7AQZQAP5VDXXV9JT77YM
max_env_step: 131072
max_train_iter: 512
num_simulations: 4
save_ckpt_after_iter: 8
summary ok: true
train_result ok: true
latest visible checkpoint: iteration_520.pth.tar
```

```text
run_id: curvytron-visual-survival-player-aware-fixed-s102-sim8-131072
attempt_id: train-gpu-l4t4-player-aware-fixed-131072x512-s102-sim8-wait-20260510
status: completed
started_at: 2026-05-10T20:32:32.404195Z
ended_at: 2026-05-10T20:36:04.324281Z
modal_task_id: ta-01KR9SEHZ0329AQ61CYRZV85WA
max_env_step: 131072
max_train_iter: 512
num_simulations: 8
save_ckpt_after_iter: 16
summary ok: true
train_result ok: true
latest visible checkpoint: iteration_540.pth.tar
```

## Verification Commands

Checked current app state:

```bash
modal app list
```

Result: CurvyZero Modal apps in the recent list were stopped, not actively
running.

Checked existing run manifests:

```bash
modal volume get curvyzero-runs training/lightzero-curvytron-visual-survival/curvytron-visual-survival-player-aware-fixed-s101-262144/run.json -
modal volume get curvyzero-runs training/lightzero-curvytron-visual-survival/curvytron-visual-survival-player-aware-fixed-s101-262144/latest_attempt.json -
modal volume get curvyzero-runs training/lightzero-curvytron-visual-survival/curvytron-visual-survival-player-aware-fixed-s101-262144/attempts/train-gpu-l4t4-player-aware-fixed-262144x1024-s101-wait-20260510/train/summary.json -
modal volume ls curvyzero-runs training/lightzero-curvytron-visual-survival/curvytron-visual-survival-player-aware-fixed-s101-262144/checkpoints/lightzero
```

Repeated the same manifest/summary/checkpoint checks for:

```text
curvytron-visual-survival-player-aware-fixed-s100-131072
curvytron-visual-survival-player-aware-fixed-s102-sim8-131072
```

## Fresh Launch

No fresh command was run.

Skipped command class:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute gpu-l4-t4 \
  --wait-for-train \
  --opponent-policy-kind fixed_straight
```

The next useful action is to evaluate the completed fixed-straight checkpoints
instead of launching another native fixed-straight training job.
