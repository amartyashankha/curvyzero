# LightZero Dummy Pong Support Scale Fix Plan - 2026-05-09

Role: custom Pong support-scale fix planner.

Scope honored: custom dummy Pong LightZero wrapper/config path only. Official
Atari configs were not touched. No training and no pytest were run.

## Finding

The Modal dummy Pong path pins `LightZero==0.2.0`. In that version, MuZero does
not consume the newer `policy.model.reward_support_range` /
`policy.model.value_support_range` fields as the decisive knobs. The decisive
v0.2.0 fields are:

- `policy.model.support_scale`, default `300`;
- `policy.model.reward_support_size`, default `601`;
- `policy.model.value_support_size`, default `601`.

I verified this against the pinned upstream tag:

- `lzero/policy/muzero.py` defines `support_scale=300` in policy model defaults
  and builds value/reward `DiscreteSupport(-support_scale, support_scale, 1)`.
- `lzero/model/muzero_model_mlp.py` defaults both categorical output heads to
  size `601`.
- The CartPole MuZero config that our dummy Pong path patches does not set these
  support fields itself.

## Why This Matters

Custom dummy Pong is a sparse, tiny-scale reward task: per-step reward is `0`,
terminal ego win is `+1`, and terminal ego loss is `-1`. A 601-way
`[-300, 300]` support is legal, but it is a poor calibration target for this
problem because most categorical atoms are irrelevant and the value/reward heads
spend capacity on output classes the task cannot naturally use.

The same issue is likely to matter for later CurvyTron transfer. CurvyTron
reward/value targets will also start from small, bounded signals: survival,
terminal win/loss, collision avoidance, and shaped safety terms. Before moving
from dummy Pong to CurvyTron-scale experiments, the custom environment path
needs proof that the compiled MuZero model is using the intended task-scale
support, not silently inheriting Atari/general-control defaults.

This is not an official Atari change. Official Atari can reasonably keep the
stock LightZero support behavior unless a separate Atari-specific audit says
otherwise. This plan only changes and verifies custom dummy Pong wrappers that
we control as a bridge toward CurvyTron.

## Patch Implemented

Small custom dummy Pong patch:

- Added an explicit optional `support_scale` argument to the config smoke, tiny
  train smoke, and scaled train attempt wrappers.
- When `support_scale` is provided, the patched dummy Pong config writes:
  - `policy.model.support_scale = support_scale`;
  - `policy.model.reward_support_size = 2 * support_scale + 1`;
  - `policy.model.value_support_size = 2 * support_scale + 1`.
- If both explicit support ranges are provided and are symmetric unit ranges
  such as `[-5, 6, 1]`, the config derives `support_scale=5` and writes the
  same v0.2.0 fields.
- The patched surface and target replay metadata now record `support_scale`,
  reward/value support sizes, ranges, and `categorical_distribution`.
- The config/import smoke now logs the compiled `cfg.policy.model` support
  fields so the decisive value is visible after DI-engine compilation.

I intentionally left `DEFAULT_SUPPORT_SCALE = None`. This keeps legacy no-scale
calls and existing 601-class checkpoint scorecard paths unchanged. Flipping the
default to `5` is likely a good next experiment, but it is not risk-free because
older checkpoints with 601-way reward/value heads need either an explicit
`support_scale=300` load path or shape inference before scoring.

## Smallest Safe Run

No training:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --mode config-import \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0 \
  --support-scale 5
```

Expected decisive fields in the returned JSON:

```json
{
  "patched_config": {
    "surface": {
      "support_scale": 5,
      "reward_support_size": 11,
      "value_support_size": 11
    }
  },
  "compiled_config": {
    "policy_model_cfg": {
      "support_scale": 5,
      "reward_support_size": 11,
      "value_support_size": 11
    }
  }
}
```

The proof artifact is the returned JSON from
`lightzero_dummy_pong_config_import_smoke`, or the same JSON captured by Modal
logs. The decisive proof fields are:

- `patched_config.surface.support_scale`
- `patched_config.surface.reward_support_size`
- `patched_config.surface.value_support_size`
- `compiled_config.policy_model_cfg.support_scale`
- `compiled_config.policy_model_cfg.reward_support_size`
- `compiled_config.policy_model_cfg.value_support_size`

The patch is proven only when the patched surface and compiled config agree. For
`--support-scale 5`, the expected compiled values are `support_scale=5` and
`reward_support_size=value_support_size=11`.

Equivalent range-driven smoke:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --mode config-import \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0 \
  --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 \
  --value-support-min -5 --value-support-max 6 --value-support-delta 1
```

## Next Decision

After the compile smoke confirms the actual compiled fields, the next safe
training command can pass `--support-scale 5` explicitly. Do not make it the
global dummy Pong default until checkpoint loaders either:

- accept a `support_scale` argument for new small-support checkpoints; or
- infer reward/value support size from checkpoint tensor shapes and set
  `support_scale=300` for legacy 601-head checkpoints.

Keep the official Atari path out of this decision. If Atari support behavior is
ever changed, track it in a separate official-Atari plan and compare against
upstream defaults directly.

## Verification

Ran syntax-only verification:

```bash
uv run python -m py_compile \
  src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py
```

Result: passed. No pytest. No LightZero training.
