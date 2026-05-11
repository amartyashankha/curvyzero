# 2026-05-09 Modal LightZero Pong Dry Config Smoke

## Question

Can the next stock LightZero sanity step inspect and cap the Atari Pong MuZero
config without starting an Atari trainer or touching ALE/Gym/ROM runtime
dependencies?

## Setup

Added a separate dry-only Modal module:

```text
src/curvyzero/infra/modal/lightzero_pong_dry_config_smoke.py
```

The module uses the same pinned CPU image shape as the earlier LightZero
smokes:

```text
LightZero==0.2.0
```

It imports `zoo.atari.config.atari_muzero_segment_config`, monkeypatches
`lzero.entry.train_muzero_segment` to capture configs, calls the stock config
builder for `PongNoFrameskip-v4`, applies tiny CPU patches to the captured
config, and returns the original and patched surfaces. It does not instantiate
Atari and does not call the trainer.

## Command

```sh
uv run python -m py_compile src/curvyzero/infra/modal/lightzero_pong_dry_config_smoke.py
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_dry_config_smoke
```

No pytest was run.

## Results

Both commands passed.

Modal result:

```text
ok: true
remote_elapsed_sec: 12.235089
packages: LightZero 0.2.0, DI-engine 0.5.3, torch 2.11.0, easydict 1.13
call_policy: dry_config_patch_only_trainer_entrypoint_monkeypatched
trainer_entrypoint: lzero.entry.train_muzero_segment
trainer_signature includes max_train_iter and max_env_step
train_result: null
```

Captured stock Pong surface:

```text
env_id: PongNoFrameskip-v4
policy_type: muzero
env_type: atari_lightzero
model_type: conv
observation_shape: [4, 96, 96]
action_space_size: 6
collector_env_num: 8
evaluator_env_num: 3
n_evaluator_episode: 3
num_simulations: 50
batch_size: 256
cuda: true
max_env_step: 500000
```

Patched dry surface:

```text
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
num_simulations: 2
batch_size: 4
update_per_collect: 1
cuda: false
max_env_step: 4
exp_name: /tmp/curvyzero-lightzero-pong-dry/seed-0
```

The import emitted LightZero/Gym dependency warnings, including Gym's
unmaintained warning and optional package warnings for numba, pyecharts, and
transformers. These did not fail the dry config smoke.

## Interpretation

This is a useful small stock-reference step. It confirms that the installed
LightZero Pong config can be captured and capped without launching the Atari
trainer.

It is not evidence that a real LightZero Pong trainer is cheap or ready. The
dry smoke deliberately avoids the hard part: ALE/Gym/EnvPool environment
creation, Atari ROM handling, replay, and the heavyweight default visual
trainer loop.

## Artifacts

- Modal run URL:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-y5j0ctBh5z092w4sXotc2C`
- Code module:
  `src/curvyzero/infra/modal/lightzero_pong_dry_config_smoke.py`

## Follow-ups

Wait on a real stock LightZero Pong trainer until there is a clear reason to
pay the Atari dependency cost. If it becomes necessary, make the next step a
separate environment-creation smoke that proves ALE/Gym/ROM availability before
calling `train_muzero_segment`.
