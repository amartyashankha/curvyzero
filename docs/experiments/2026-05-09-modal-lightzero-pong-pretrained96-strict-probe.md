# 2026-05-09 Modal LightZero Pong Pretrained96 Strict Probe

## Question

Can the public Hugging Face `OpenDILabCommunity/PongNoFrameskip-v4-MuZero`
checkpoint strict-load and run a dry forward when we recreate its older
`[4, 96, 96]`, `downsample=True` LightZero Atari config surface?

## Clarification

This lane is not here because LightZero lacks MuZero. LightZero has MuZero and
the repo already has a current 64x64 stock Atari MuZero lane. This separate
pretrained96 lane exists because the public pretrained checkpoint matches a
different LightZero/DI-engine model/config surface than the current 64x64 stock
config.

## Setup

Added separate pretrained96 wrappers:

```text
src/curvyzero/infra/modal/lightzero_pong_pretrained96_checkpoint_probe.py
src/curvyzero/infra/modal/lightzero_pong_pretrained96_eval_smoke.py
```

The checkpoint probe deliberately uses the older segment config surface:

```text
module: zoo.atari.config.atari_muzero_segment_config
env_id: PongNoFrameskip-v4
env_type: atari_lightzero
policy_type: muzero
model_type: conv
observation_shape: [4, 96, 96]
action_space_size: 6
downsample: true
```

The current 64x64 path remains separate in:

```text
src/curvyzero/infra/modal/lightzero_pong_checkpoint_probe.py
src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py
```

## Commands

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_pretrained96_checkpoint_probe.py src/curvyzero/infra/modal/lightzero_pong_pretrained96_eval_smoke.py
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_pretrained96_checkpoint_probe
```

No pytest was run. No training was run.

## Result

Local compile passed. The strict-load/forward gate did not pass, so the
no-fallback eval was not run.

Modal run:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-HbwLRvdNQyNWH5cpMArk5u
```

Probe artifact:

```text
training/lightzero-official-visual-pong-pretrained96/pretrained/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/probe/lightzero_visual_pong_pretrained96_strict_load_20260509T201008Z.json
sha256: 905fe4045912b8acdd8744ec770496a5057ab16f658d3c69f785aee035abb2d5
```

Checkpoint:

```text
training/lightzero-official-visual-pong/pretrained/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/pytorch_model.bin
sha256: f59d3b43c0ed3d70a60c10acf0a427c7576d2109edb095802cbc9384efc7b4bf
bytes: 73821201
```

Status:

```text
checkpoint_load_ok: true
state_dict_ok: true
surface_ok: true
strict_direct_model_load_ok: false
strict_policy_model_load_ok: false
direct_forward_ok: false
```

The recreated config surface was correct:

```text
observation_shape: [4, 96, 96]
action_space_size: 6
downsample: true
policy_type: muzero
env_type: atari_lightzero
```

Exact blocker:

```text
Error(s) in loading state_dict for MuZeroModel:
  Unexpected key(s) in state_dict:
  "representation_network.downsample_net.conv2.weight".
```

The direct forward was skipped because strict load failed.

## Interpretation

The first 64x64 blocker was resolved: the `[32, 576]` versus `[32, 1024]` and
`[1024, 2304]` versus `[1024, 4096]` size mismatches disappear when using the
older `[4, 96, 96]`, `downsample=True` surface.

The remaining blocker is a narrower source-version drift: the public HF
checkpoint contains one extra `representation_network.downsample_net.conv2.weight`
tensor that the current `LightZero==0.2.0` 96x96 `MuZeroModel` does not register.
That is still a strict-load failure, so no eval is valid from this wrapper.

Do not eval this checkpoint with non-strict load, tensor deletion, model
fallback, or the current 64x64 path.
