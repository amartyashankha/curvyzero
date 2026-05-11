# 2026-05-09 Modal LightZero Pong Checkpoint Load Smoke

## Question

Can the mirrored stock visual Atari Pong `iteration_1.pth.tar` checkpoint from
the tiny trainer be loaded back into the same LightZero Atari MuZero
model/policy/config surface before any longer training?

## Setup

Added a dedicated loader wrapper:

```text
src/curvyzero/infra/modal/lightzero_pong_checkpoint_probe.py
```

It uses the same official stock visual Pong surface as the tiny trainer:

```text
source module: zoo.atari.config.atari_muzero_config
entrypoint lineage: lzero.entry.train_muzero
env id: PongNoFrameskip-v4
env type: atari_lightzero
model type: conv
model class: lzero.model.muzero_model.MuZeroModel
policy class: lzero.policy.muzero.MuZeroPolicy
observation shape: [4, 64, 64]
action space size: 6
```

This probe does not score gameplay. It only loads checkpoint tensors, records
strict/partial load status, reports model shape, and runs one cheap direct
zero-observation `initial_inference`.

## Commands

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_checkpoint_probe.py src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py
```

```sh
uv run --extra modal python -c "import curvyzero.infra.modal.lightzero_pong_checkpoint_probe as m; print(m.DEFAULT_CHECKPOINT_REF)"
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_checkpoint_probe
```

No pytest was run.

## Results

Compile passed. Import check passed.

Modal checkpoint-load smoke passed:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-YqqMmryhwgtKdFTbStnDKJ
ok: true
remote_elapsed_sec: 15.895763
checkpoint_load_ok: true
state_dict_ok: true
strict_direct_model_load_ok: true
strict_policy_model_load_ok: true
partial_direct_model_load_ok: false
partial_policy_model_load_ok: false
direct_forward_ok: true
```

Checkpoint ref:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/checkpoints/lightzero/iteration_1.pth.tar
sha256: bf82857b2c74ba96072738cc745e2c141e182802e37e9d37cacb09172cf5e931
bytes: 96204713
```

Output artifact:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T171834Z-1798cd6bef57/attempts/attempt-20260509T171834Z-fd4b5559bec6/probe/lightzero_visual_pong_checkpoint_load_20260509T172430Z.json
sha256: 7aad9fbf9abac812d5ffe723cb0a02a291867101c38e67d19ff56bf3ca7fd91b
```

Model/forward signal:

```text
state_dict path: model
tensor_count: 175
direct model load: strict true, candidate as_is, missing [], unexpected []
policy model load: strict true, candidate as_is, missing [], unexpected []
model parameters: 7,997,608
input shape: [1, 4, 64, 64]
latent_state shape: [1, 64, 8, 8]
policy_logits shape: [1, 6]
value shape: [1, 601]
greedy zero-frame action id: 0
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

## Interpretation

The checkpoint-load gate is positive. The tiny trainer's mirrored
`iteration_1.pth.tar` can be read from the Modal Volume, its `model` state dict
loads strictly into both the direct conv `MuZeroModel` and the compiled
`MuZeroPolicy` model, and a direct forward produces the expected visual MuZero
output shapes.

This is still not a gameplay or quality result. It only proves the stock visual
checkpoint bytes round-trip through the matching LightZero config/model/policy
surface.

## Follow-up

The next step is a tiny eval-only smoke that uses the loaded policy on one
short capped ALE Pong episode or a single LightZero `eval_mode.forward` call
with a real reset observation. Do not start a longer training run until that
eval-only path is proven.
