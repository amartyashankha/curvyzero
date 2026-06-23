# LightZero model parity inventory, 2026-05-23f

Scope: repo/package inspection only. I did not launch or modify any live Coach or LightZero training run. I instantiated the installed LightZero model on CPU with zero tensors only to confirm shapes and `state_dict` names.

## Answer

CurvyTron stock LightZero training uses DI-engine policy type `muzero`, implemented by `lzero.policy.muzero.MuZeroPolicy` from installed `LightZero==0.2.0` / `DI-engine==0.5.3`. For the active/default CurvyTron stock lane, `policy.model.model_type` is `conv`, so `MuZeroPolicy.default_model()` resolves to PyTorch class:

```text
lzero.model.muzero_model.MuZeroModel
```

The CurvyTron stock launcher/config path is:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
  -> lz_config.build_visual_survival_configs(...)
  -> create_config.policy = {type: "muzero", import_names: ["lzero.policy.muzero"]}
  -> lzero.entry.train_muzero([main_config, create_config], ...)
```

Default stock CurvyTron env/model surface:

```text
env_variant: source_state_fixed_opponent
env type: curvyzero_source_state_visual_survival_lightzero
env id: CurvyZeroSourceStateVisualSurvivalLightZero-v0
observation_shape: [4, 64, 64]
action_space_size: 3
reward_variant: sparse_outcome
opponent_policy_kind: fixed_straight
learner_seat_mode: random_per_episode
collector_env_num: 256
n_episode: 256
num_simulations: 8
train batch_size: 64
```

Effective model config observed from the builder plus `MuZeroModel` defaults:

```text
model_type: conv
observation_shape: [4, 64, 64]
image_channel: 4
frame_stack_num: 1
gray_scale: True
action_space_size: 3
downsample: True
norm_type: BN
discrete_action_encoding_type: one_hot
self_supervised_learning_loss: True
categorical_distribution: True
support_scale: 1
reward_support_size: 3
value_support_size: 3
reward_support_range: (-1.0, 2.0, 1.0)
value_support_range: (-1.0, 2.0, 1.0)
num_res_blocks: 1
num_channels: 64
reward_head_channels: 16
value_head_channels: 16
policy_head_channels: 16
reward/value/policy head hidden_channels: [32]
projection: 4096 -> 1024 -> 1024 -> 1024
prediction_head: 1024 -> 512 -> 1024
```

Note: the Atari template leaves `use_sim_norm: True` and `use_sim_norm_kl_loss: False` in `policy.model`, but installed `MuZeroModel.__init__` accepts those via `**kwargs` and does not pass `use_sim_norm` into `RepresentationNetwork` on this path. The observed instantiated state has no `sim_norm` entries.

## Inference Shapes

For batch size `B`, `initial_inference(obs)` receives:

```text
obs: torch.float32 [B, 4, 64, 64]
```

It returns `MZNetworkOutput(value, reward, policy_logits, latent_state)`:

```text
value: torch.float32 [B, 3]
reward: Python list length B of scalar 0.0 values
policy_logits: torch.float32 [B, 3]
latent_state: torch.float32 [B, 64, 8, 8]
```

For recurrent search, `recurrent_inference(latent_state, action)` receives:

```text
latent_state: torch.float32 [B, 64, 8, 8]
action: torch integer [B] or [B, 1], values in {0, 1, 2}
```

With one-hot action encoding, the dynamics network concatenates action planes and internally sees:

```text
state_action_encoding: torch.float32 [B, 67, 8, 8]
```

It returns:

```text
value: torch.float32 [B, 3]
reward: torch.float32 [B, 3]
policy_logits: torch.float32 [B, 3]
latent_state: torch.float32 [B, 64, 8, 8]  # next latent state
```

In `MuZeroPolicy.collect/eval`, LightZero unpacks as `(latent_state, reward, value, policy_logits)`, inverse-transforms `value` to scalar roots for MCTS, converts `latent_state` to numpy, and passes `reward_roots` as the initial zero list.

## JAX Conversion Inventory

For checkpoint conversion, the relevant trained network tensors live under `checkpoint["model"]`. LightZero checkpoints also include `checkpoint["target_model"]`, `checkpoint["optimizer"]`, `last_iter`, and `last_step`; the target model is a second copy for resume fidelity, while inference/search parity only needs `model`. CurvyZero frozen-opponent loading already tolerates `model.`, `_learn_model.`, and `module.` prefixes.

Convert trainable parameter arrays and BatchNorm state. `*.num_batches_tracked` are PyTorch int64 BatchNorm counters; a JAX inference model can usually ignore them, but exact training-resume parity should preserve equivalent BN counter/state if the JAX BN implementation uses it.

Full observed `MuZeroModel.state_dict()` key inventory for the stock default config:

```text
representation_network.downsample_net.conv1.weight (32, 4, 3, 3) float32
representation_network.downsample_net.norm1.weight (32,) float32
representation_network.downsample_net.norm1.bias (32,) float32
representation_network.downsample_net.norm1.running_mean (32,) float32
representation_network.downsample_net.norm1.running_var (32,) float32
representation_network.downsample_net.norm1.num_batches_tracked () int64
representation_network.downsample_net.resblocks1.0.conv1.0.weight (32, 32, 3, 3) float32
representation_network.downsample_net.resblocks1.0.conv1.1.weight (32,) float32
representation_network.downsample_net.resblocks1.0.conv1.1.bias (32,) float32
representation_network.downsample_net.resblocks1.0.conv1.1.running_mean (32,) float32
representation_network.downsample_net.resblocks1.0.conv1.1.running_var (32,) float32
representation_network.downsample_net.resblocks1.0.conv1.1.num_batches_tracked () int64
representation_network.downsample_net.resblocks1.0.conv2.0.weight (32, 32, 3, 3) float32
representation_network.downsample_net.resblocks1.0.conv2.1.weight (32,) float32
representation_network.downsample_net.resblocks1.0.conv2.1.bias (32,) float32
representation_network.downsample_net.resblocks1.0.conv2.1.running_mean (32,) float32
representation_network.downsample_net.resblocks1.0.conv2.1.running_var (32,) float32
representation_network.downsample_net.resblocks1.0.conv2.1.num_batches_tracked () int64
representation_network.downsample_net.downsample_block.conv1.0.weight (64, 32, 3, 3) float32
representation_network.downsample_net.downsample_block.conv1.1.weight (64,) float32
representation_network.downsample_net.downsample_block.conv1.1.bias (64,) float32
representation_network.downsample_net.downsample_block.conv1.1.running_mean (64,) float32
representation_network.downsample_net.downsample_block.conv1.1.running_var (64,) float32
representation_network.downsample_net.downsample_block.conv1.1.num_batches_tracked () int64
representation_network.downsample_net.downsample_block.conv2.0.weight (64, 64, 3, 3) float32
representation_network.downsample_net.downsample_block.conv2.1.weight (64,) float32
representation_network.downsample_net.downsample_block.conv2.1.bias (64,) float32
representation_network.downsample_net.downsample_block.conv2.1.running_mean (64,) float32
representation_network.downsample_net.downsample_block.conv2.1.running_var (64,) float32
representation_network.downsample_net.downsample_block.conv2.1.num_batches_tracked () int64
representation_network.downsample_net.downsample_block.conv3.0.weight (64, 32, 3, 3) float32
representation_network.downsample_net.resblocks2.0.conv1.0.weight (64, 64, 3, 3) float32
representation_network.downsample_net.resblocks2.0.conv1.1.weight (64,) float32
representation_network.downsample_net.resblocks2.0.conv1.1.bias (64,) float32
representation_network.downsample_net.resblocks2.0.conv1.1.running_mean (64,) float32
representation_network.downsample_net.resblocks2.0.conv1.1.running_var (64,) float32
representation_network.downsample_net.resblocks2.0.conv1.1.num_batches_tracked () int64
representation_network.downsample_net.resblocks2.0.conv2.0.weight (64, 64, 3, 3) float32
representation_network.downsample_net.resblocks2.0.conv2.1.weight (64,) float32
representation_network.downsample_net.resblocks2.0.conv2.1.bias (64,) float32
representation_network.downsample_net.resblocks2.0.conv2.1.running_mean (64,) float32
representation_network.downsample_net.resblocks2.0.conv2.1.running_var (64,) float32
representation_network.downsample_net.resblocks2.0.conv2.1.num_batches_tracked () int64
representation_network.downsample_net.resblocks3.0.conv1.0.weight (64, 64, 3, 3) float32
representation_network.downsample_net.resblocks3.0.conv1.1.weight (64,) float32
representation_network.downsample_net.resblocks3.0.conv1.1.bias (64,) float32
representation_network.downsample_net.resblocks3.0.conv1.1.running_mean (64,) float32
representation_network.downsample_net.resblocks3.0.conv1.1.running_var (64,) float32
representation_network.downsample_net.resblocks3.0.conv1.1.num_batches_tracked () int64
representation_network.downsample_net.resblocks3.0.conv2.0.weight (64, 64, 3, 3) float32
representation_network.downsample_net.resblocks3.0.conv2.1.weight (64,) float32
representation_network.downsample_net.resblocks3.0.conv2.1.bias (64,) float32
representation_network.downsample_net.resblocks3.0.conv2.1.running_mean (64,) float32
representation_network.downsample_net.resblocks3.0.conv2.1.running_var (64,) float32
representation_network.downsample_net.resblocks3.0.conv2.1.num_batches_tracked () int64
representation_network.resblocks.0.conv1.0.weight (64, 64, 3, 3) float32
representation_network.resblocks.0.conv1.1.weight (64,) float32
representation_network.resblocks.0.conv1.1.bias (64,) float32
representation_network.resblocks.0.conv1.1.running_mean (64,) float32
representation_network.resblocks.0.conv1.1.running_var (64,) float32
representation_network.resblocks.0.conv1.1.num_batches_tracked () int64
representation_network.resblocks.0.conv2.0.weight (64, 64, 3, 3) float32
representation_network.resblocks.0.conv2.1.weight (64,) float32
representation_network.resblocks.0.conv2.1.bias (64,) float32
representation_network.resblocks.0.conv2.1.running_mean (64,) float32
representation_network.resblocks.0.conv2.1.running_var (64,) float32
representation_network.resblocks.0.conv2.1.num_batches_tracked () int64
dynamics_network.conv.weight (64, 67, 3, 3) float32
dynamics_network.norm_common.weight (64,) float32
dynamics_network.norm_common.bias (64,) float32
dynamics_network.norm_common.running_mean (64,) float32
dynamics_network.norm_common.running_var (64,) float32
dynamics_network.norm_common.num_batches_tracked () int64
dynamics_network.resblocks.0.conv1.0.weight (64, 64, 3, 3) float32
dynamics_network.resblocks.0.conv1.1.weight (64,) float32
dynamics_network.resblocks.0.conv1.1.bias (64,) float32
dynamics_network.resblocks.0.conv1.1.running_mean (64,) float32
dynamics_network.resblocks.0.conv1.1.running_var (64,) float32
dynamics_network.resblocks.0.conv1.1.num_batches_tracked () int64
dynamics_network.resblocks.0.conv2.0.weight (64, 64, 3, 3) float32
dynamics_network.resblocks.0.conv2.1.weight (64,) float32
dynamics_network.resblocks.0.conv2.1.bias (64,) float32
dynamics_network.resblocks.0.conv2.1.running_mean (64,) float32
dynamics_network.resblocks.0.conv2.1.running_var (64,) float32
dynamics_network.resblocks.0.conv2.1.num_batches_tracked () int64
dynamics_network.conv1x1_reward.weight (16, 64, 1, 1) float32
dynamics_network.conv1x1_reward.bias (16,) float32
dynamics_network.norm_reward.weight (16,) float32
dynamics_network.norm_reward.bias (16,) float32
dynamics_network.norm_reward.running_mean (16,) float32
dynamics_network.norm_reward.running_var (16,) float32
dynamics_network.norm_reward.num_batches_tracked () int64
dynamics_network.fc_reward_head.0.weight (32, 1024) float32
dynamics_network.fc_reward_head.0.bias (32,) float32
dynamics_network.fc_reward_head.1.weight (32,) float32
dynamics_network.fc_reward_head.1.bias (32,) float32
dynamics_network.fc_reward_head.1.running_mean (32,) float32
dynamics_network.fc_reward_head.1.running_var (32,) float32
dynamics_network.fc_reward_head.1.num_batches_tracked () int64
dynamics_network.fc_reward_head.3.weight (3, 32) float32
dynamics_network.fc_reward_head.3.bias (3,) float32
prediction_network.resblocks.0.conv1.0.weight (64, 64, 3, 3) float32
prediction_network.resblocks.0.conv1.1.weight (64,) float32
prediction_network.resblocks.0.conv1.1.bias (64,) float32
prediction_network.resblocks.0.conv1.1.running_mean (64,) float32
prediction_network.resblocks.0.conv1.1.running_var (64,) float32
prediction_network.resblocks.0.conv1.1.num_batches_tracked () int64
prediction_network.resblocks.0.conv2.0.weight (64, 64, 3, 3) float32
prediction_network.resblocks.0.conv2.1.weight (64,) float32
prediction_network.resblocks.0.conv2.1.bias (64,) float32
prediction_network.resblocks.0.conv2.1.running_mean (64,) float32
prediction_network.resblocks.0.conv2.1.running_var (64,) float32
prediction_network.resblocks.0.conv2.1.num_batches_tracked () int64
prediction_network.conv1x1_value.weight (16, 64, 1, 1) float32
prediction_network.conv1x1_value.bias (16,) float32
prediction_network.conv1x1_policy.weight (16, 64, 1, 1) float32
prediction_network.conv1x1_policy.bias (16,) float32
prediction_network.norm_value.weight (16,) float32
prediction_network.norm_value.bias (16,) float32
prediction_network.norm_value.running_mean (16,) float32
prediction_network.norm_value.running_var (16,) float32
prediction_network.norm_value.num_batches_tracked () int64
prediction_network.norm_policy.weight (16,) float32
prediction_network.norm_policy.bias (16,) float32
prediction_network.norm_policy.running_mean (16,) float32
prediction_network.norm_policy.running_var (16,) float32
prediction_network.norm_policy.num_batches_tracked () int64
prediction_network.fc_value.0.weight (32, 1024) float32
prediction_network.fc_value.0.bias (32,) float32
prediction_network.fc_value.1.weight (32,) float32
prediction_network.fc_value.1.bias (32,) float32
prediction_network.fc_value.1.running_mean (32,) float32
prediction_network.fc_value.1.running_var (32,) float32
prediction_network.fc_value.1.num_batches_tracked () int64
prediction_network.fc_value.3.weight (3, 32) float32
prediction_network.fc_value.3.bias (3,) float32
prediction_network.fc_policy.0.weight (32, 1024) float32
prediction_network.fc_policy.0.bias (32,) float32
prediction_network.fc_policy.1.weight (32,) float32
prediction_network.fc_policy.1.bias (32,) float32
prediction_network.fc_policy.1.running_mean (32,) float32
prediction_network.fc_policy.1.running_var (32,) float32
prediction_network.fc_policy.1.num_batches_tracked () int64
prediction_network.fc_policy.3.weight (3, 32) float32
prediction_network.fc_policy.3.bias (3,) float32
projection.0.weight (1024, 4096) float32
projection.0.bias (1024,) float32
projection.1.weight (1024,) float32
projection.1.bias (1024,) float32
projection.1.running_mean (1024,) float32
projection.1.running_var (1024,) float32
projection.1.num_batches_tracked () int64
projection.3.weight (1024, 1024) float32
projection.3.bias (1024,) float32
projection.4.weight (1024,) float32
projection.4.bias (1024,) float32
projection.4.running_mean (1024,) float32
projection.4.running_var (1024,) float32
projection.4.num_batches_tracked () int64
projection.6.weight (1024, 1024) float32
projection.6.bias (1024,) float32
projection.7.weight (1024,) float32
projection.7.bias (1024,) float32
projection.7.running_mean (1024,) float32
projection.7.running_var (1024,) float32
projection.7.num_batches_tracked () int64
prediction_head.0.weight (512, 1024) float32
prediction_head.0.bias (512,) float32
prediction_head.1.weight (512,) float32
prediction_head.1.bias (512,) float32
prediction_head.1.running_mean (512,) float32
prediction_head.1.running_var (512,) float32
prediction_head.1.num_batches_tracked () int64
prediction_head.3.weight (1024, 512) float32
prediction_head.3.bias (1024,) float32
```

Key conversion implications:

- PyTorch conv weights are `[out_channels, in_channels, kernel_h, kernel_w]`; typical JAX/Flax conv kernels are `[kernel_h, kernel_w, in_channels, out_channels]`.
- PyTorch linear weights are `[out_features, in_features]`; transpose for a JAX dense kernel `[in_features, out_features]`.
- BatchNorm parameters map as `weight -> scale`, `bias -> bias`, `running_mean -> mean`, `running_var -> var`; preserve epsilon/momentum semantics separately in the JAX implementation.
- The initial-inference reward is not a learned tensor; reproduce LightZero's root reward as zeros/list-equivalent for root preparation.
- The support size is 3 for current default sparse stock CurvyTron. Dense/reward-bonus variants can increase `support_scale` and therefore the value/reward head output sizes; a converter should read `policy.model.reward_support_size` and `value_support_size` from the checkpoint sidecar/config, not hard-code 3.

## Evidence

- `pyproject.toml`: optional `lightzero` dependency pins `LightZero==0.2.0` and `torch==2.8.0`.
- `src/curvyzero/training/lightzero_config_builder.py`: stock visual-survival builder imports `zoo.atari.config.atari_muzero_config`, sets `create_config.policy.type = "muzero"`, `policy.model.model_type = "conv"`, `image_channel = 4`, `frame_stack_num = 1`, `observation_shape = [4, 64, 64]`, and `action_space_size` from the env spec.
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`: source-state fixed-opponent env config exposes `observation_shape = STACKED_SOURCE_STATE_GRAY64_SHAPE` and `action_space_size = ACTION_COUNT`.
- `src/curvyzero/env/observation_surface_contract.py`: `POLICY_STACK_SHAPE = (4, 64, 64)`.
- Installed `lzero/policy/muzero.py`: `MuZeroPolicy.default_model()` maps `model_type == "conv"` to `MuZeroModel` from `lzero.model.muzero_model`; `_state_dict_learn()` saves `model`, `target_model`, and `optimizer`.
- Installed `lzero/model/muzero_model.py`: `MuZeroModel.initial_inference` and `recurrent_inference` implementations and defaults.
- Installed `ding/worker/learner/learner_hook.py`: `SaveCkptHook` writes `engine.policy.state_dict()` plus `last_iter` and `last_step`.
