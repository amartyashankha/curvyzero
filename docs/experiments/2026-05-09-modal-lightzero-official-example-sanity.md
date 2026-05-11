# Modal LightZero Official Example Sanity

Date: 2026-05-09.

Purpose: prove LightZero itself can run a tiny official/simple trainer lane on
Modal before attributing dummy Pong issues to Pong-specific integration.

## Example Choice

Inspected `/tmp/lightzero-src` and the existing Modal lanes under
`src/curvyzero/infra/modal/`.

Chosen example: stock LightZero CartPole MuZero:

```text
zoo.classic_control.cartpole.config.cartpole_muzero_config
lzero.entry.train_muzero
```

Reason: it is the smallest official trainer path already covered by our Modal
code. The official Atari/Pong path exists, but it routes through the heavier
visual Atari segment config and is less appropriate for a cheap sanity lane.

## Commands

Compile only:

```sh
uv run python -m py_compile src/curvyzero/infra/modal/lightzero_cartpole_tiny_train_smoke.py
```

Tiny Modal train smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke \
  --mode train \
  --run-id official-cartpole-sanity-20260509 \
  --attempt-id attempt-train-smoke-1
```

No pytest was run.

## Modal Run

Run URL:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-5XQeVIVi0jqM0pKKTxLKh6
```

Run volume refs:

```text
summary: training/lightzero-official-cartpole/official-cartpole-sanity-20260509/attempts/attempt-train-smoke-1/train/summary.json
training_signals: training/lightzero-official-cartpole/official-cartpole-sanity-20260509/attempts/attempt-train-smoke-1/train/lightzero_training_signals.json
artifacts_manifest: training/lightzero-official-cartpole/official-cartpole-sanity-20260509/attempts/attempt-train-smoke-1/train/lightzero_artifacts_manifest.json
checkpoint_manifest: training/lightzero-official-cartpole/official-cartpole-sanity-20260509/checkpoints/lightzero/manifest.json
```

Mirrored checkpoints:

```text
training/lightzero-official-cartpole/official-cartpole-sanity-20260509/checkpoints/lightzero/ckpt_best.pth.tar
training/lightzero-official-cartpole/official-cartpole-sanity-20260509/checkpoints/lightzero/iteration_0.pth.tar
training/lightzero-official-cartpole/official-cartpole-sanity-20260509/checkpoints/lightzero/iteration_1.pth.tar
```

## Result

Status: passed.

```text
ok: true
status: completed
mode: train
remote_elapsed_sec: 18.017989
train_result.ok: true
return_type: MuZeroPolicy
train elapsed_sec: 6.325154
problems: []
```

Package surface:

```text
LightZero 0.2.0
DI-engine 0.5.3
torch 2.11.0
easydict 1.13
```

Patched official CartPole surface:

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

Signals:

```text
training_iterations: [0]
checkpoint_iterations: [0, 1]
final_rewards: [9.0]
eval_episode_return_mean: 9.0
envstep_count: 9.0
total_loss_avg: 44.759914
policy_loss_avg: 3.119163
value_loss_avg: 38.391567
target_reward_avg: 0.75
```

Interpretation: the official/simple LightZero trainer lane works on Modal CPU.
It imports the stock CartPole example, calls `lzero.entry.train_muzero`, returns
a `MuZeroPolicy`, emits learner/evaluator metrics, writes LightZero artifacts,
and mirrors checkpoints into the run volume. This is not a policy-quality claim
and does not validate the heavier official Atari/Pong trainer, but it does
separate "LightZero can train at all on Modal" from Pong-specific failures.
