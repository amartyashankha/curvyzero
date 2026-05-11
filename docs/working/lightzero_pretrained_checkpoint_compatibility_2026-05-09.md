# LightZero Pretrained Checkpoint Compatibility - 2026-05-09

Owner lane: pretrained checkpoint compatibility research. This note only
answers whether the OpenDILab/Hugging Face Pong pretrained checkpoint should be
used through an older 96x96 config recreation, or whether there is a matching
pretrained checkpoint for the current 64x64 stock LightZero Atari config.

## Decision

Yes path: use the OpenDILabCommunity `PongNoFrameskip-v4-MuZero`
pretrained checkpoint only with the Hugging Face model-card config surface:
`observation_shape` / `obs_shape` `[4, 96, 96]`, `action_space_size` `6`,
`downsample=True`, `env.type=atari_lightzero`, `policy.type=muzero`, and the
model-card package era (`Gym 0.25.1`, `DI-engine v0.5.0`, `PyTorch
2.0.1+cu117` as reported by the card). Do not force that checkpoint through
the current repo's 64x64 stock config.

No path: I did not find a public OpenDILabCommunity Hugging Face pretrained
MuZero checkpoint matching the current 64x64 stock LightZero Atari MuZero
config. The current 64x64 lane has local tiny/from-scratch checkpoints that
strict-load, but those are not pretrained policy-quality checkpoints.

Practical answer: recreate/select the older 96x96/downsample config for the HF
pretrained Pong checkpoint. For the current 64x64 config, keep using locally
trained current-surface checkpoints or train a new one; do not expect the HF
MuZero checkpoint to fit.

## Compatibility Evidence

The HF `OpenDILabCommunity/PongNoFrameskip-v4-MuZero` model card is explicitly
for LightZero/DI-engine MuZero on `PongNoFrameskip-v4`. Its usage loads
`pytorch_model.bin`, reads `policy_config.py`, constructs `MuZeroAgent`, and
reports the config as:

```text
env.obs_shape: [4, 96, 96]
policy.model.observation_shape: [4, 96, 96]
policy.model.action_space_size: 6
policy.model.downsample: True
policy.type: muzero
env.type: atari_lightzero
reported mean_reward: 20.4 +/- 0.49
last update: 2023-12-15
```

The current upstream LightZero Atari MuZero config is a different surface:

```text
zoo/atari/config/atari_muzero_config.py
env.observation_shape: (4, 64, 64)
policy.model.observation_shape: (4, 64, 64)
policy.model.downsample: True
policy.model.model_type: conv
policy.model.use_sim_norm: True
```

Local repo evidence already tested the wrong pairing. The
`lightzero_stock_replication_control_plan_2026-05-09.md` note records a strict
load attempt of the HF `pytorch_model.bin` into the current 64x64 config. It
loaded the file and found a state dict, but strict direct-model and policy-model
loads both failed. The recorded shape blockers were:

```text
Unexpected key:
representation_network.downsample_net.conv2.weight

Size mismatches:
dynamics_network.fc_reward_head.0.weight checkpoint [32, 576] vs current [32, 1024]
prediction_network.fc_value.0.weight checkpoint [32, 576] vs current [32, 1024]
prediction_network.fc_policy.0.weight checkpoint [32, 576] vs current [32, 1024]
projection.0.weight checkpoint [1024, 2304] vs current [1024, 4096]
```

Those sizes line up with an older 96x96/downsample spatial contract, not the
current 64x64 contract. Current LightZero model source still contains explicit
64-vs-96 shape handling in `DownSample` / `PredictionNetwork`, so the right fix
is matching the model config, not non-strict loading or tensor surgery.

## Checkpoint Search Result

Primary Hugging Face API search for `author=OpenDILabCommunity` and
`PongNoFrameskip-v4` returned these relevant repos:

```text
PongNoFrameskip-v4-MuZero
PongNoFrameskip-v4-EfficientZero
PongNoFrameskip-v4-SampledEfficientZero
PongNoFrameskip-v4-C51
PongNoFrameskip-v4-DQN
PongNoFrameskip-v4-PPO
PongNoFrameskip-v4-PPOOffPolicy
PongNoFrameskip-v4-IMPALA
```

The MuZero-family pretrained repos checked (`MuZero`, `EfficientZero`,
`SampledEfficientZero`) all expose 96x96 policy configs. The older DI-engine
non-MuZero repos checked are 84x84 or other non-MuZero policy surfaces, so they
are not matching replacements for the current 64x64 LightZero MuZero stock
config.

## Go / No-Go

Go: evaluate or deploy `OpenDILabCommunity/PongNoFrameskip-v4-MuZero` through
the downloaded `policy_config.py` / recreated 96x96 downsample config, with the
Atari wrapper producing `[4, 96, 96]` observations. Prefer the model-card
dependency era if current LightZero APIs drift.

No-go: strict-load the HF pretrained `pytorch_model.bin` into the current
`zoo.atari.config.atari_muzero_config` 64x64 harness. No-go for non-strict load,
fallback model actions, or weight-shape surgery as proof of compatibility.

No-go: claim there is a public OpenDILabCommunity 64x64 pretrained Pong MuZero
checkpoint until a repo/file is found whose config says `[4, 64, 64]` and
`policy.type=muzero`.

## Sources

- Hugging Face model card, `OpenDILabCommunity/PongNoFrameskip-v4-MuZero`:
  https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero
- Hugging Face raw config:
  https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/raw/main/policy_config.py
- Hugging Face model API listing for files and metadata:
  https://huggingface.co/api/models/OpenDILabCommunity/PongNoFrameskip-v4-MuZero
- Hugging Face author search used for alternate checkpoint scan:
  https://huggingface.co/api/models?author=OpenDILabCommunity&search=PongNoFrameskip-v4&limit=100
- Current upstream LightZero Atari MuZero config:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_config.py
- Current upstream LightZero Atari MuZero segment config:
  https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_segment_config.py
- Current upstream LightZero model source shape handling:
  https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/model/common.py
- Local prior strict-load failure note:
  `docs/working/lightzero_stock_replication_control_plan_2026-05-09.md`
- Local current 64x64 strict-load success for locally trained checkpoint:
  `docs/experiments/2026-05-09-modal-lightzero-pong-checkpoint-load-smoke.md`

No code was changed. No pytest was run.
