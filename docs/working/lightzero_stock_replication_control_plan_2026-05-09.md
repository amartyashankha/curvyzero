# LightZero Stock Replication Control Plan - 2026-05-09

Scope: stock LightZero Atari Pong control only. No code changes and no pytest.

## Question

Can we strict-load an official/pretrained Pong checkpoint and use it as the next
stock LightZero control?

## Local Source Search

- `third_party/` has only `curvytron-reference`; no vendored LightZero checkout.
- Local LightZero stock Pong wrappers already target:
  - `zoo.atari.config.atari_muzero_config`
  - `lzero.entry.train_muzero`
  - `PongNoFrameskip-v4`
  - `atari_lightzero`
  - visual conv MuZero with 6 Atari actions
- Existing local docs show our stock Atari Pong lane can train, mirror, strict-load,
  and no-fallback eval locally trained checkpoints.
- Existing local docs also identify the OpenDILab Hugging Face model card
  `OpenDILabCommunity/PongNoFrameskip-v4-MuZero` as the only concrete official-ish
  pretrained Pong MuZero artifact.

## Primary Sources Checked

- LightZero quick start says stock Pong MuZero runs with:
  `python3 -u zoo/atari/config/atari_muzero_config.py`
  Source: https://opendilab.github.io/LightZero/tutorials/installation/installation_and_quickstart.html
- Upstream LightZero `main` stock Atari config defaults to `PongNoFrameskip-v4`,
  `atari_lightzero`, `train_muzero`, `num_simulations=50`, and
  `observation_shape=(4, 64, 64)`.
  Source: https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_config.py
- OpenDILabCommunity Hugging Face model card provides pretrained files:
  `.gitattributes`, `README.md`, `policy_config.py`, `pytorch_model.bin`,
  `replay.mp4`; reports mean reward `20.4 +/- 0.49`.
  Source: https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero
- Hugging Face API source metadata:
  `sha=fe6094aa2f870e99ece5a55c960724864519faa9`,
  `lastModified=2023-12-15T15:54:22Z`,
  `pytorch_model.bin` present.
  Source: https://huggingface.co/api/models/OpenDILabCommunity/PongNoFrameskip-v4-MuZero

## Pretrained Probe

Downloaded:

```text
https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/resolve/main/pytorch_model.bin
sha256 f59d3b43c0ed3d70a60c10acf0a427c7576d2109edb095802cbc9384efc7b4bf
```

Uploaded to Modal Volume:

```text
training/lightzero-official-visual-pong/pretrained/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/pytorch_model.bin
```

Strict-load probe command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_checkpoint_probe --checkpoint-ref training/lightzero-official-visual-pong/pretrained/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/pytorch_model.bin --run-id lz-visual-pong-pretrained-opendilab-hf --attempt-id strict-load-pretrained-hf --output-ref training/lightzero-official-visual-pong/pretrained/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/probe/lightzero_visual_pong_pretrained_strict_load.json
```

Probe artifact:

```text
training/lightzero-official-visual-pong/pretrained/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/probe/lightzero_visual_pong_pretrained_strict_load.json
sha256 775ab8e615098facd9f9c25f313ebd54f02395772aad2b45a5639c5f81081960
Modal app ap-FdFrLtyomoxo0EaL3YaLgJ
```

Result:

```text
checkpoint_load_ok: true
state_dict_ok: true
strict_direct_model_load_ok: false
strict_policy_model_load_ok: false
direct_forward_ok: false
```

Exact blocker:

```text
Unexpected key:
representation_network.downsample_net.conv2.weight

Size mismatches:
dynamics_network.fc_reward_head.0.weight checkpoint [32, 576] vs current [32, 1024]
prediction_network.fc_value.0.weight checkpoint [32, 576] vs current [32, 1024]
prediction_network.fc_policy.0.weight checkpoint [32, 576] vs current [32, 1024]
projection.0.weight checkpoint [1024, 2304] vs current [1024, 4096]
```

Plain read: the official/pretrained Hugging Face checkpoint is from a different
stock Atari model surface than the current LightZero stock config used by the
repo. The model-card `policy_config.py` is the older 96x96/downsample artifact;
current upstream/current image stock Atari config builds a 64x64 model. That is
not a runtime eval problem. It is a model-shape incompatibility, so no strict
eval command is valid from the current stock-config loader.

## Answer

No: we cannot strict-load the official/pretrained OpenDILab Pong checkpoint into
the current stock LightZero Atari Pong config/eval harness today.

The exact blocker is architecture/config drift between the Hugging Face
pretrained `pytorch_model.bin` and the current `zoo.atari.config.atari_muzero_config`
model shape.

Do not eval it with non-strict load, shape surgery, or fallback actions.

## Concrete Next Step

Closest-upstream bounded rung: from-scratch stock Atari Pong, not pretrained.

Use the existing current-stock config and keep it honest:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --compute gpu-l4-t4 --mode train --max-env-step 4096 --max-train-iter 32 --collector-env-num 2 --evaluator-env-num 1 --num-simulations 10 --batch-size 32 --update-per-collect 2 --max-episode-steps 512 --game-segment-length 64 --run-id lz-visual-pong-4096-sim10-s0 --attempt-id train-4096-sim10-b32-env2
```

Then eval only existing checkpoints from that run with strict no-fallback eval:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_16.pth.tar --run-id lz-visual-pong-4096-sim10-s0 --attempt-id train-4096-sim10-b32-env2 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_16_sim10_eval256/lightzero_visual_pong_eval_iteration16_sim10_eval256.json --max-env-step 4096 --max-train-iter 32 --collector-env-num 2 --evaluator-env-num 1 --num-simulations 10 --batch-size 32 --update-per-collect 2 --max-episode-steps 512 --game-segment-length 64 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback
```

Bound: this is still not official-scale quality. It is the smallest credible
current-upstream rung because it moves toward stock collector count, search,
batch, and segment length while preserving the strict-load/no-fallback rule. Do
not climb to sim25/sim50 unless this rung gives a curve signal.
