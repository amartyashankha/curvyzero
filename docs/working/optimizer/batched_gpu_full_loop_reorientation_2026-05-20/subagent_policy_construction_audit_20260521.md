# LightZero Policy Construction Audit

Date: 2026-05-21

Scope: audit only. No source code or live training state changed. The worktree was
already dirty, including the training module and optimizer docs, so treat this as
shared state.

## Short Answer

The smallest safe real-consumer canary is a new profile-only Modal function that
does not call `_run_visual_survival_train` or `lzero.entry.train_muzero`. Build the
same CurvyTron MuZero config as the trusted trainer, compile it, instantiate a
scratch `lzero.policy.muzero.MuZeroPolicy`, then call
`policy.collect_mode.forward(...)` directly on a flattened root batch.

Use:

- `curvyzero.training.lightzero_config_builder.build_visual_survival_configs`
- `ding.config.compile_config`
- `lzero.policy.muzero.MuZeroPolicy`
- `policy.collect_mode.forward(...)`

For an input stack shaped `[B,2,4,64,64]`, LightZero's actual policy consumer
expects a root batch shaped `[N,4,64,64]`. Flatten the row/player axes to
`N=B*2`, preserve `policy_env_row` and `policy_player` side metadata, and pass
`ready_env_id=np.arange(N)`.

## Existing Live Construction Path

The live trainer imports the shared config builder as `lz_config` in
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:106`.
Its Modal image installs `LightZero==0.2.0`, `jax[cuda12]`, `numpy`, `cloudpickle`,
and `pillow`, sets `PYTHONPATH=/repo/src`, and copies `src` plus the bonus sprite
sheet at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:761`.
The image and volumes are defined through `:777` and `:778`.

The live training path imports `lzero.entry`, selects the entrypoint through the
exploration-bonus helper, and resolves `train_muzero` at
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5336`.
It then calls the local wrapper `_build_visual_survival_configs(...)` at `:5342`,
compiles a summary at `:5487`, installs multiple train-time hooks, and finally
calls `train_muzero([main_config, create_config], ...)` at `:5723`.

Do not reuse `_run_visual_survival_train` for this canary. It writes run manifests,
attempt state, heartbeats, and volume artifacts before the LightZero call
(`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5306`
through `:5335`), and its profile mode can still touch run IDs, auto-resume, and
trainer hooks.

## Config Builder Facts

The config builder uses the stock LightZero Atari MuZero template:
`src/curvyzero/training/lightzero_config_builder.py:1212`.

It creates the DI-engine create config with CurvyTron env registration and
`policy: {type: "muzero", import_names: ["lzero.policy.muzero"]}` at
`src/curvyzero/training/lightzero_config_builder.py:1258` through `:1266`.

It patches the policy to the CurvyTron model surface:

- `policy.cuda`, `policy.multi_gpu`, env type, collector/evaluator counts, episode
  counts, simulations, and batch size: `src/curvyzero/training/lightzero_config_builder.py:1269`
  through `:1282`
- conv model, `image_channel=4`, `frame_stack_num=1`,
  `self_supervised_learning_loss=True`, observation shape, and action count:
  `src/curvyzero/training/lightzero_config_builder.py:1283` through `:1300`
- env observation shape, `gray_scale=True`, `image_channel=4`, action-space,
  reward target metadata, and policy observation contract:
  `src/curvyzero/training/lightzero_config_builder.py:1304` through `:1348`

The public wrapper is `build_visual_survival_configs(...)` at
`src/curvyzero/training/lightzero_config_builder.py:1478`, which normalizes args
into `VisualSurvivalConfigSpec` and returns `main_config`, `create_config`,
`surface`, and `patches` at `:1530`.

The source-state policy stack shape is `[4,64,64]`: `POLICY_STACK_SHAPE` is defined
from four gray64 frames in `src/curvyzero/env/observation_surface_contract.py:35`,
and `STACKED_SOURCE_STATE_GRAY64_SHAPE = POLICY_STACK_SHAPE` is wired in
`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:204`.
The env reports `lightzero_payload_shape` and `model_observation_shape` from this
same shape at `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:2004`
through `:2011`.

## Existing Scratch/Direct Policy Precedent

The closest already-working pattern is the checkpoint eval helper:
`src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:_make_policy_and_env`.
It imports `compile_config`, `get_vec_env_setting`, and `MuZeroPolicy` at
`src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:724`,
builds CurvyTron visual survival configs at `:728`, compiles at `:808`, forces
`cfg.policy.device` and `cfg.policy.cuda` at `:815` through `:821`, and
instantiates `MuZeroPolicy(cfg.policy)` at `:823`. For the canary, borrow this
policy-construction core but omit checkpoint loading and env creation.

There is also a scratch random-policy precedent in
`src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py:340`.
It imports `compile_config` and `MuZeroPolicy`, patches an Atari MuZero config,
compiles it, sets `cfg.policy.cuda/device`, instantiates `MuZeroPolicy(cfg.policy)`,
and puts `_model` in eval mode at `:363` through `:392`. This is useful precedent,
but it is the older stacked-debug surface, not the current source-state fixed
opponent path.

The frozen-opponent provider has a smaller direct loader:
`src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:377`. It builds
configs in `_lightzero_policy_configs` at `:508`, compiles at `:419`, sets device
at `:426`, and instantiates `MuZeroPolicy(cfg.policy)` at `:429`. This is compact,
but it hardcodes the legacy stacked-debug env/type and should not be the source of
truth for the current canary.

## Actual Collect Consumer Calls Already In Repo

Direct real collect-mode calls already exist:

- Background GIF path: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:9879`
  through `:9895`
- Tournament policy action path: `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3544`
  through `:3560`
- Two-seat smoke batched path: `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2738`
  through `:2755`

These all pass a Torch `obs_tensor` on the policy device plus NumPy
`action_mask`, Python/list `to_play`, NumPy `ready_env_id`, `temperature`, and
`epsilon`.

## Recommended Canary Skeleton

Inside a profile-only Modal function:

```python
import copy
from pathlib import Path

import numpy as np
import torch
from ding.config import compile_config
from lzero.policy.muzero import MuZeroPolicy

from curvyzero.training import lightzero_config_builder as lz_config
```

Build with the current source-state defaults, but use tiny run-independent paths:

```python
patched = lz_config.build_visual_survival_configs(
    seed=seed,
    exp_name=Path("/tmp/curvyzero_real_consumer_canary_exp"),
    telemetry_path=Path("/tmp/curvyzero_real_consumer_canary_env_steps.jsonl"),
    cuda=True,
    max_env_step=1,
    source_max_steps=source_max_steps,
    decision_ms=DEFAULT_DECISION_MS,
    collector_env_num=batch_size,
    evaluator_env_num=1,
    n_evaluator_episode=1,
    n_episode=batch_size,
    num_simulations=num_simulations,
    batch_size=batch_size,
    lightzero_eval_freq=0,
    lightzero_multi_gpu=False,
    max_train_iter=1,
    save_ckpt_after_iter=1_000_000,
    env_variant=DEFAULT_ENV_VARIANT,
    reward_variant=DEFAULT_REWARD_VARIANT,
    reward_outcome_alpha=DEFAULT_REWARD_OUTCOME_ALPHA,
    ego_action_straight_override_probability=0.0,
    control_noise_profile_id="none",
    disable_death_for_profile=False,
    env_telemetry_stride=1,
    env_manager_type="base",
    opponent_policy_kind=DEFAULT_OPPONENT_POLICY_KIND,
    opponent_use_cuda=False,
    opponent_checkpoint=None,
    opponent_snapshot_ref=None,
    opponent_checkpoint_state_key=None,
)
cfg = compile_config(
    copy.deepcopy(patched["main_config"]),
    seed=seed,
    auto=True,
    create_cfg=copy.deepcopy(patched["create_config"]),
    save_cfg=False,
)
cfg.policy.cuda = True
cfg.policy.device = "cuda"
policy = MuZeroPolicy(cfg.policy)
policy._model.eval()
```

Then consume the pre-scalar stack:

```python
# stack: torch uint8 or float tensor shaped [B, 2, 4, 64, 64] on CUDA.
B = int(stack.shape[0])
flat = stack.reshape(B * 2, 4, 64, 64)
if flat.dtype == torch.uint8:
    obs_tensor = flat.to(dtype=torch.float32).div_(255.0)
else:
    obs_tensor = flat.to(dtype=torch.float32)

action_mask_np = np.ones((B * 2, 3), dtype=np.float32)
ready_env_id = np.arange(B * 2)
to_play = [-1] * (B * 2)

with torch.no_grad():
    output = policy.collect_mode.forward(
        obs_tensor,
        action_mask=action_mask_np,
        temperature=1.0,
        to_play=to_play,
        epsilon=0.0,
        ready_env_id=ready_env_id,
    )
```

If the profile surface already has real masks, use `action_mask.reshape(B*2, 3)`
instead of an all-true mask. Keep all timing and output decoding inside the
profile function; do not write checkpoints, manifests, or run status files.

## Package And Modal Setup

The local `pyproject.toml` only declares the `modal` extra at `pyproject.toml:21`;
LightZero/DI-engine are runtime image dependencies, not local project deps.

For a Modal function in the existing trainer module, reuse `image=image` from
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:761`.
That image already has LightZero 0.2.0, Torch via LightZero dependencies, JAX CUDA,
NumPy, source code, and the sprite-sheet file. For a pure return-value canary, no
Modal volume is required. If the function is placed in the existing training app
and uses existing constants, `TRAINER_VOLUMES` is available at
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:786`,
but mounting volumes is unnecessary unless the canary writes artifacts.

For GPU selection, reuse the existing compute constants:
`LIGHTZERO_VERSION`, `DEFAULT_NUM_SIMULATIONS`, and `DEFAULT_BATCH_SIZE` are defined
at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:379`
and `:398` through `:399`; cheap GPU resources are defined at `:669`.

## Caveats

- `collect_mode.forward` is real LightZero MuZero policy/search, but the current
  codebase call sites pass NumPy action masks and Python/NumPy IDs. The first
  canary can honestly test GPU-resident observation handoff while leaving mask/ID
  residency as a separate gate.
- Do not pass `[B,2,4,64,64]` directly to `collect_mode.forward`; flatten to
  `[B*2,4,64,64]` and carry row/player metadata for decode and legality checks.
- A scratch random policy should not load a checkpoint or call `_make_policy_and_env`
  as-is, because that helper requires a checkpoint state dict and creates an env
  at `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:827`
  through `:840`.
- The builder may instantiate a tiny local env to discover source-state fixed
  opponent wrapper fields at `src/curvyzero/training/lightzero_config_builder.py:167`
  through `:178`. That is not a live run, but it is not a zero-side-effect pure
  import.
- Avoid the live `profile` mode of `_run_visual_survival_train` for this canary:
  it still goes through run-management plumbing and can auto-resume or install
  trainer hooks before `train_muzero`.
