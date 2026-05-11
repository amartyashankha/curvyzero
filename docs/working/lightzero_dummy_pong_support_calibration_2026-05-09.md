# LightZero Dummy Pong Support Calibration - 2026-05-09

Role: reward/value support calibration auditor.

Scope: custom dummy Pong LightZero config/code, local dummy Pong reward path, and
LightZero source for the pinned package used by the Modal image. No training and
no pytest were run.

## Short Answer

The custom dummy Pong environment gives LightZero only sparse game reward:
`0` during play, `+1` for the scoring ego player, and `-1` for the losing ego
player. The shaped loss-delay value is written to telemetry only. I did not find
it being returned as the environment reward.

The support-range knobs are suspicious. Our code can record
`reward_support_range` and `value_support_range` in the patched config surface,
but the Modal image installs `LightZero==0.2.0`, and that version's MuZero source
uses `policy.model.support_scale`, default `300`, not those range fields.

So yes: `support_scale=300` is still compiled somewhere for the version we pin,
unless DI-engine config compilation rewrites it in a way this local machine
cannot verify. The cheapest falsifying check is a dry config/import probe that
prints the compiled `cfg.policy.model.support_scale`,
`reward_support_size`, and `value_support_size`, not just our patched surface.

## What Is Known

- The pinned Modal package is `LightZero==0.2.0` in
  `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`.
- Our dummy Pong support CLI defaults are all `None`, so no support override
  happens unless the caller passes all three min/max/delta values.
- When support override values are provided, our code writes
  `policy.model.reward_support_range` and `policy.model.value_support_range`.
  It does not write `policy.model.support_scale`, `reward_support_size`, or
  `value_support_size`.
- The train wrapper records only the patched surface fields
  `reward_support_range` and `value_support_range` in
  `patched_terminal_target_config`.
- The real env reward returned to LightZero is `pong_step.rewards[ego_agent]`.
  In `PongEnv`, this is sparse terminal score reward only.
- `shaped_loss_delay_return` is computed inside `info["curvyzero_pong"]` at
  episode end. It is telemetry, not the returned reward.

## Version-Specific LightZero Finding

This is the crux.

In current LightZero `main`, MuZero has moved to explicit
`reward_support_range` and `value_support_range` fields. But the repo pins
`LightZero==0.2.0`, and the `v0.2.0` source still has:

- `policy.model.support_scale=300` in `lzero/policy/muzero.py`.
- `DiscreteSupport(-support_scale, support_scale, delta=1)` for both value and
  reward in `_init_learn`.
- `InverseScalarTransform(support_scale, ...)` for converting value logits back
  to scalar values.
- `MuZeroModelMLP` takes `reward_support_size=601` and
  `value_support_size=601`, not support ranges.

That means adding `reward_support_range=(-5, 6, 1)` can be harmless metadata in
`LightZero==0.2.0` unless something also changes the fields that v0.2.0
actually consumes.

Sources checked:

- Local config patch:
  `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- Local train wrapper:
  `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- Local env reward path:
  `src/curvyzero/training/dummy_pong.py`
  and `src/curvyzero/training/lightzero_dummy_pong_env.py`
- Pinned upstream:
  `https://github.com/opendilab/LightZero/blob/v0.2.0/lzero/policy/muzero.py`
  and
  `https://github.com/opendilab/LightZero/blob/v0.2.0/lzero/model/muzero_model_mlp.py`
- Current upstream contrast:
  `https://github.com/opendilab/LightZero/blob/main/lzero/policy/muzero.py`

## Could This Flatten Small Rewards?

Not in the simple sense of dividing `+1` by `300`.

LightZero first applies the MuZero scalar transform. For target `+1`, the
transformed scalar is about `+0.415`; for `-1`, about `-0.415`. With unit support
atoms, the categorical target still lives around atoms `0` and `+1`, or `-1`
and `0`.

But a `[-300, 300]` support still makes the head predict 601 classes for a task
whose real reward/value scale is about `[-1, 1]`. That can make learning a tiny
sparse signal harder than necessary:

- most classes are never useful for dummy Pong;
- initial softmax mass is spread across 601 atoms;
- reward/value gradients are aimed at two nearby atoms inside a very wide head;
- MCTS value conversion also assumes that wide support when reading logits.

So "flatten" is the wrong exact word, but the suspicion is real: the output head
and inverse transform are badly calibrated for tiny sparse Pong.

For shaped telemetry, the current code does not train on it. If a future branch
does train on `shaped_loss_delay_return`, it has the same small-scale issue
because shaped values also sit near `[-1, 1]`.

## What Is Suspicious

1. Our summaries can say `reward_support_range=[-5, 6, 1]` and
   `value_support_range=[-5, 6, 1]`, while the actual pinned LightZero policy may
   still build `support_scale=300`.

2. Existing docs already noticed this mismatch. The new audit agrees with that
   warning.

3. The local repo has no importable `lzero`, `ding`, or `zoo` package, so I
   could not locally instantiate the compiled DI-engine policy config.

4. The code records the patched support range but does not record
   `support_scale`, `reward_support_size`, or `value_support_size`. That hides
   the decisive fields.

## Tiny Local Probe Run

I ran a tiny local import probe:

```text
lzero: ModuleNotFoundError
ding: ModuleNotFoundError
zoo.classic_control.cartpole.config.cartpole_muzero_config: ModuleNotFoundError
```

I also ran a tiny dummy Pong reward probe with `PongEnv(max_steps=8)` and
constant `stay` actions. It returned `0.0` every step and then truncated at step
8 with total rewards `0.0` for both players. This matches the sparse reward
contract.

## Cheapest Falsifying Check

Do not train. Do not pytest.

Add one tiny dry inspection path to the existing config/import smoke, or run the
same logic inline inside its dry Modal function:

```python
cfg = compile_config(main_config, seed=seed, env=None, auto=True, create_cfg=create_config, save_cfg=False)
print({
    "support_scale": cfg.policy.model.get("support_scale"),
    "reward_support_range": cfg.policy.model.get("reward_support_range"),
    "value_support_range": cfg.policy.model.get("value_support_range"),
    "reward_support_size": cfg.policy.model.get("reward_support_size"),
    "value_support_size": cfg.policy.model.get("value_support_size"),
})
```

Falsifies the concern if the compiled dry config shows no `support_scale=300`
and shows small support sizes/ranges that the v0.2.0 policy actually uses.

Confirms the concern if it shows `support_scale=300` and/or
`reward_support_size=value_support_size=601` while our patched surface says
`[-5, 6, 1]`.

If confirmed, the smallest real fix is to patch v0.2.0 fields directly:

- set `policy.model.support_scale` to a small integer, probably `5`;
- set `policy.model.reward_support_size` and `policy.model.value_support_size`
  to the matching size if required by the model default path;
- record all three decisive fields in `patched_terminal_target_config`.

