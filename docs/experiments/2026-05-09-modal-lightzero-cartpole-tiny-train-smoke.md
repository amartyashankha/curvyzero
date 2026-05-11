# Modal LightZero CartPole Tiny Train Smoke

Date: 2026-05-09.

Purpose: prepare and run the next stock LightZero replication step after the
LightZero config smoke. This is stock LightZero CartPole MuZero, not
CurvyZero's trainer and not Pong.

## Commands

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode train
```

```sh
uv run python -m py_compile src/curvyzero/infra/modal/lightzero_cartpole_tiny_train_smoke.py
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode progression
```

No pytest was run.

## Result

Both runs passed on Modal CPU.

```text
dry smoke: ok true, remote_elapsed_sec 12.847235
train smoke: ok true, remote_elapsed_sec 13.269896
train_result: ok true, return_type MuZeroPolicy, elapsed_sec 4.356072
```

Package surface:

```text
LightZero 0.2.0
DI-engine 0.5.3
torch 2.11.0
easydict 1.13
```

Patched stock CartPole surface:

```text
env_id: CartPole-v0
policy_type: muzero
env_type: cartpole_lightzero
model_type: mlp
action_space_size: 2
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
n_episode: 1
num_simulations: 2
batch_size: 4
update_per_collect: 1
cuda: false
max_train_iter: 1
max_env_step: 4
```

The trainer entrypoint was `lzero.entry.train_muzero`; its signature included
both `max_train_iter` and `max_env_step`.

## Interpretation

The tiny real trainer started, ran one initial evaluation episode, ran one
learner update, saved temporary checkpoints inside the remote container, and
returned a `MuZeroPolicy`.

The initial evaluator episode logged `reward 9.0` and `envstep_count 9.0`,
which is above the requested `max_env_step: 4`. Treat that as LightZero
finishing a small evaluator episode before the trainer loop observes the cap,
not as permission to loosen caps. The run is a stock-entrypoint smoke only; it
does not prove policy quality.

## Progression Follow-up

Question: can we run a slightly longer but still cheap stock CartPole MuZero
Modal job and record real progress signals, not just a trainer return value?

Implementation change: `lightzero_cartpole_tiny_train_smoke.py` now supports
`--mode progression`. The progression mode stays on CPU and patches the
installed stock CartPole config to:

```text
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
n_episode: 1
num_simulations: 5
batch_size: 16
update_per_collect: 4
eval_freq: 1
cuda: false
max_train_iter: 4
max_env_step: 128
```

It captures the trainer stderr/stdout stream, parses evaluator and learner
tables, records checkpoint-save signals, and scans both LightZero artifact
roots. The second root is necessary because this LightZero version writes some
checkpoint/log paths under `.//tmp/...` while the formatted config files land
under `/tmp/...`.

Result on Modal CPU, 2026-05-09:

```text
ok: true
remote_elapsed_sec: 15.603642
train_result.ok: true
return_type: MuZeroPolicy
train elapsed_sec: 5.651327
packages: LightZero 0.2.0, DI-engine 0.5.3, torch 2.11.0, easydict 1.13
```

Progress signals:

```text
final_rewards: [33.0]
checkpoint_iterations: [0, 4]
max_checkpoint_iteration: 4
reward_mean: 33.0
eval_episode_return_mean: 33.0
envstep_count: 33.0
total_loss_avg: 45.577473
policy_loss_avg: 3.855631
value_loss_avg: 38.391567
target_reward_avg: 0.927083
```

Artifacts observed inside the remote container:

```text
/tmp/curvyzero-lightzero-cartpole-tiny/seed-0/formatted_total_config.py
/tmp/curvyzero-lightzero-cartpole-tiny/seed-0/total_config.py
tmp/curvyzero-lightzero-cartpole-tiny/seed-0/ckpt/ckpt_best.pth.tar
tmp/curvyzero-lightzero-cartpole-tiny/seed-0/ckpt/iteration_0.pth.tar
tmp/curvyzero-lightzero-cartpole-tiny/seed-0/ckpt/iteration_4.pth.tar
tmp/curvyzero-lightzero-cartpole-tiny/seed-0/log/evaluator/evaluator_logger.txt
tmp/curvyzero-lightzero-cartpole-tiny/seed-0/log/learner/learner_logger.txt
tmp/curvyzero-lightzero-cartpole-tiny/seed-0/log/serial/events.out.tfevents.1778300611.modal
```

Modal run URL:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-ZVLCMheMhxmAtgJsgjOMII
```

Interpretation: this validates the existing-example lane for stock LightZero
CartPole MuZero progression. It is a real stock trainer progression signal:
the trainer ran through update/checkpoint index 4, emitted evaluator and
learner metrics, and wrote checkpoint/log artifacts. It is not a policy-quality
claim and does not validate stock Pong training.
