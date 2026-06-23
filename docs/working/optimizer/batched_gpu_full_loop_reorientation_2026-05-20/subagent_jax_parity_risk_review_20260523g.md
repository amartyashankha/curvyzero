# JAX Parity Risk Review, 2026-05-23g

Scope: sidecar review only. I did not edit code, start training, touch Modal, or
inspect live runs. This note is about the smallest PyTorch LightZero MuZero ->
JAX shadow-model parity proof needed before the MCTX/JAX search lane can claim
it is using the real CurvyTron policy.

## Current LightZero Model Facts

The trusted CurvyTron builder starts from `zoo.atari.config.atari_muzero_config`
and patches a copied config. It sets `create_config.policy.type="muzero"` with
`lzero.policy.muzero`, and patches `policy.model.model_type="conv"`,
`image_channel=4`, `frame_stack_num=1`, `self_supervised_learning_loss=True`,
`observation_shape=list(env_spec["observation_shape"])`, and the env action
count. See `src/curvyzero/training/lightzero_config_builder.py:1257` and
`src/curvyzero/training/lightzero_config_builder.py:1265` through
`src/curvyzero/training/lightzero_config_builder.py:1301`.

For the current source-state policy surface, that means:

- model class: installed `lzero.model.muzero_model.MuZeroModel`
- policy class: installed `lzero.policy.muzero.MuZeroPolicy`
- model type: `conv`
- observation: float32 `[B, 4, 64, 64]`
- action count: `3`
- root latent after representation: `[B, 64, 8, 8]`
- recurrent action encoding: one-hot action planes, so dynamics input is
  `[B, 67, 8, 8]`

The policy observation contract confirms the 4-frame `64x64` controlled-player
view, model dtype, value range, render modes, and CPU default backend in
`src/curvyzero/env/observation_surface_contract.py:35` through
`src/curvyzero/env/observation_surface_contract.py:68`.

Checkpoint loading should reuse
`load_lightzero_curvytron_visual_survival_policy(...)`. It strict-loads
`policy._model`, validates observation metadata, infers reward/value support
sizes from checkpoint head weights, and calls `model.eval()`. See
`src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:399` through
`src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:454`, plus
support inference at `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:457`.

## Risky Parity Layers

This is not just conv/dense transposition.

- Downsample path: `RepresentationNetwork` uses `DownSample` plus residual
  blocks before the main representation residual block. Shape `64 -> 8` depends
  on the 64-pixel special case. See installed
  `.venv/lib/python3.11/site-packages/lzero/model/muzero_model.py:122` and
  `.venv/lib/python3.11/site-packages/lzero/model/common.py:601`.
- BatchNorm everywhere: representation, dynamics, reward head, value head,
  policy head, projection, and prediction head all carry running mean/var. The
  harness must run Torch in eval mode and map BN state, not just weights.
- Recurrent action encoding: `_dynamics` mutates action shape, casts to long,
  scatters one-hot rows, expands them over the latent grid, and concatenates on
  channel axis. See installed
  `.venv/lib/python3.11/site-packages/lzero/model/muzero_model.py:331` through
  `.venv/lib/python3.11/site-packages/lzero/model/muzero_model.py:369`.
- Dynamics residual add: the dynamics conv output is normalized, then adds the
  previous latent before ReLU. A missing residual add can still produce plausible
  logits while killing recurrent parity. See installed
  `.venv/lib/python3.11/site-packages/lzero/model/muzero_model.py:517`.
- Root reward is special: `initial_inference` returns a Python list of zeros,
  not reward-head logits. The JAX root function must reproduce that contract
  separately from recurrent reward logits. See installed
  `.venv/lib/python3.11/site-packages/lzero/model/muzero_model.py:232`.
- SSL projection heads exist because the builder sets
  `self_supervised_learning_loss=True`. They are not needed for MCTS inference,
  but the converter must account for those state_dict keys so "converted all
  required weights" is not silently false. See installed
  `.venv/lib/python3.11/site-packages/lzero/model/muzero_model.py:187` through
  `.venv/lib/python3.11/site-packages/lzero/model/muzero_model.py:207`.
- Support size is not safely hard-coded. Current sparse default may be size 3,
  but reward variants and RND can expand support ranges. The checkpoint loader
  already infers this from `fc_reward_head.3.weight` and `fc_value.3.weight`.

## Smallest Confidence Gates

1. Config gate: print the exact copied LightZero model config and fail unless
   it is `conv`, `[4,64,64]`, `A=3`, `categorical_distribution=True`,
   `self_supervised_learning_loss=True`, and support sizes match the checkpoint
   heads.
2. Key coverage gate: every required inference key is converted exactly once;
   SSL-only keys may be labeled "unused by MCTS inference" but must not vanish
   without a count.
3. Layer-order gate: compare intermediate tensors, not only final actions:
   representation latent, prediction policy/value, recurrent next latent,
   recurrent reward, recurrent policy/value.
4. Input fixture gate: use zero, ramp, checkerboard, one-hot frame, and seeded
   random observations. Use actions `[0,1,2]` in the same batch so one-hot plane
   bugs show up.
5. BatchNorm gate: run Torch `eval()` and JAX eval mode, then report BN
   running mean/var coverage. A train-mode Torch comparison is not useful.
6. Root contract gate: assert initial reward is list/zero-equivalent and
   recurrent reward is logits with the checkpoint support size.
7. Tolerance gate: start CPU-only with `atol=1e-4, rtol=1e-4`; promote GPU only
   after CPU passes. Report max-abs and max-rel by output name.
8. Safety label gate: every report says `profile_only=true`,
   `not_train_muzero=true`, `not_mctx=true`, `touches_live_runs=false`, and
   `trainer_defaults_changed=false`.

## Hidden Old-Path Traps

- The frozen-opponent helper currently builds a legacy debug env id in
  `_lightzero_policy_configs`, while the current builder defaults to the
  source-state fixed-opponent lane. It is probably okay for model-only loading
  because the tensor is still `[4,64,64]`, but the parity report should print
  both the checkpoint metadata and the exact model config so this does not look
  like source-state env parity. See
  `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:536`.
- Old docs and checkpoints mention `[4,64,64]` debug occupancy and source-state
  surfaces interchangeably. The checkpoint metadata validator is the protection;
  do not bypass it.
- The existing MCTX compact service is synthetic and explicitly
  `profile_only_jax_mctx_gumbel_muzero_search_not_lightzero_ctree`, so green
  MCTX legality/profile rows do not prove real-model parity. See
  `src/curvyzero/training/mctx_compact_search_service.py:1`.
- MCTX uses Gumbel MuZero policy semantics in the current sidecar, while stock
  LightZero collect uses its own MuZero CTree path. Raw model parity must come
  before any search-result comparison.
- Any "latest checkpoint" ref is unsafe for this harness. Use immutable
  `iteration_N.pth.tar` or a local fixed path; otherwise the parity report can
  race a live checkpoint writer.

Bottom line: the smallest useful proof is not search parity. It is a raw
`MuZeroModel.initial_inference` and `recurrent_inference` parity harness with
strict checkpoint load, full required-key accounting, BN state mapping, and
intermediate tensor checks.
