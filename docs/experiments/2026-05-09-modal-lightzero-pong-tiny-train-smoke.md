# 2026-05-09 Modal LightZero Pong Tiny Train Smoke

## Question

Can the stock LightZero visual Atari Pong path run one brutally capped Modal
trainer smoke after the ROM-enabled env gate passed?

## Setup

Added a separate tiny trainer wrapper:

```text
src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py
```

It uses the shared AutoROM image helper:

```text
src/curvyzero/infra/modal/lightzero_atari_rom_image.py
```

The wrapper uses the official stock visual MuZero Pong path:

```text
source module: zoo.atari.config.atari_muzero_config
entrypoint: lzero.entry.train_muzero
env id: PongNoFrameskip-v4
env type: atari_lightzero
model type: conv
action space size: 6
```

This is infrastructure-only. It is not a quality run and should not be read as
evidence that the policy learned Pong.

## Caps

Default train-mode caps:

```text
cpu: 1
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
n_episode: 1
num_simulations: 2
batch_size: 4
update_per_collect: 1
cuda: false
max_env_step: 4
max_train_iter: 1
collect_max_episode_steps: 64
eval_max_episode_steps: 64
game_segment_length: 16
save_ckpt_after_iter: 1
```

The 64-step episode caps are intentional. Without them, `train_muzero` still
collects full episodes even when `max_env_step` is tiny.

## Commands

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py src/curvyzero/infra/modal/lightzero_pong_env_smoke.py src/curvyzero/infra/modal/lightzero_pong_dry_config_smoke.py
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --mode train
```

No pytest was run.

## Results

Compile passed.

Dry Modal run passed:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-zJylQsu1IPoOIbtlhDlO2P
ok: true
mode: dry
trainer_entrypoint: lzero.entry.train_muzero
patched surface: PongNoFrameskip-v4, muzero, atari_lightzero, conv,
  action_space_size 6, collector/evaluator env 1,
  num_simulations 2, batch_size 4, update_per_collect 1,
  cuda false, max_env_step 4, max_train_iter 1,
  collect/eval max episode steps 64, game_segment_length 16
train_result: null
```

Train Modal run passed:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-MbyIGvX6R815WMZcYzcAyu
ok: true
status: completed
mode: train
return_type: MuZeroPolicy
remote_elapsed_sec: 30.227448
train_result.elapsed_sec: 12.329494
training_iterations: [0]
final_rewards: [-1.0]
problems: []
```

Package surface:

```text
LightZero 0.2.0
DI-engine 0.5.3
torch 2.11.0
gym 0.25.1
gymnasium 0.28.0
ale-py 0.8.1
opencv-python-headless 4.11.0.86
AutoROM 0.6.1
```

LightZero wrote a tiny experiment under `/tmp/curvyzero-lightzero-visual-pong/...`
inside the Modal container. The wrapper mirrored discovered checkpoints and
persisted manifests/log tails/configs to the existing `curvyzero-runs` Modal
Volume.

## Artifacts

Run ids:

```text
run_id: lz-visual-pong-20260509T171834Z-1798cd6bef57
attempt_id: attempt-20260509T171834Z-fd4b5559bec6
```

Primary refs:

```text
summary:
training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/train/summary.json

training signals:
training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/train/lightzero_training_signals.json

artifact manifest:
training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/train/lightzero_artifacts_manifest.json

checkpoint mirror manifest:
training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/manifest.json
```

Mirrored checkpoints:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/ckpt_best.pth.tar
sha256: 078b718718223cf2cbfd4c2f9905575edca48d682b2830c692f0f5ecd3065027

training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_0.pth.tar
sha256: d1170a00db2687f59c6d7530050deaf78416ec08563acbd1774a54f606b8ed67

training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_1.pth.tar
sha256: bf82857b2c74ba96072738cc745e2c141e182802e37e9d37cacb09172cf5e931
```

## Interpretation

The stock visual Atari Pong trainer gate is now positive: the ROM-enabled Modal
image can call LightZero `train_muzero` on the official conv Atari Pong config,
collect/evaluate through ALE, run one learner update, and write checkpoint
artifacts.

This does not say anything useful about policy quality. The reward `-1.0` and
single training iteration are expected for a mechanical smoke. The useful fact
is that the official visual training stack is no longer blocked by imports,
ROMs, config capture, environment reset/step, or basic artifact handling.

## Follow-up

The exact next step after this trainer smoke is a checkpoint-load smoke for the
mirrored `iteration_1.pth.tar` using the same stock Atari config surface. Do
not start a longer training run until checkpoint loading and a tiny eval-only
path are proven.
