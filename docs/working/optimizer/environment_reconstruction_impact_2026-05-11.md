# Environment Reconstruction Impact On Optimizer

Date: 2026-05-11

Status: optimizer working note after reviewing the live Environment docs and
the current CurvyTron LightZero trainer imports.

2026-05-15 correction: current trusted guidance is stock LightZero
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train`
with `env_variant=source_state_fixed_opponent` and frozen-opponent route docs.
The old custom `--mode two-seat-selfplay` launcher is historical evidence only.

## Plain Answer

The fixed/frozen-opponent CurvyTron LightZero stock-control/profile path is not
isolated from Environment Reconstruction.

It uses:

```text
env_variant=source_state_fixed_opponent
-> CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv
-> VectorMultiplayerEnv(batch_size=1, player_count=2)
-> SourceStateGray64Renderer
-> wrapper-owned [4,64,64] visual stack
-> LightZero train_muzero
```

So background changes to `VectorMultiplayerEnv` reset, step, terminal
state, source-state arrays, or source-state visual rendering can affect
optimizer profiles and stock-control runs.

## What This Is Not

This is not automatically full CurvyTron fidelity.

The active runtime under hardening is `VectorMultiplayerEnv`. It has stronger
source-backed 2P behavior than the old toy/debug paths, and it now has a
source-state visual LightZero route proof, but the Optimizer should still report
the exact runtime surface instead of claiming broad game fidelity. Known limits:

- fixed-opponent single-ego for the stock-control LightZero trainer;
- not two-seat current-policy self-play;
- not browser/canvas pixel fidelity;
- natural source-default bonus spawn is enabled in the current source-state
  profile path, but fast-runtime catch/effect support is not complete yet;
- not broad 3P/4P trainer-ready replay;
- not a full environment-fidelity claim.

The current visual tensor is better than the old debug occupancy smoke because
it renders from `VectorMultiplayerEnv.state` through the
Environment-owned `curvyzero_source_state_gray64/v0` schema. It is still a
source-state geometry raster, not a browser pixel claim.

## What Environment Changes Can Break

High-risk shared pieces:

- `VectorMultiplayerEnv.reset(...)`
- `VectorMultiplayerEnv.step(...)`
- terminal/done/truncation/final-observation behavior
- `vector_runtime.step_many(...)`
- reset/spawn/warmup RNG helpers used by the default 2P no-bonus path
- `SourceStateGray64Renderer` state-field requirements and grayscale semantics

No longer isolated from the current optimizer trainer:

- natural source-default bonus spawn, because the current source-state wrapper
  constructs `VectorMultiplayerEnv(..., natural_bonus_spawn=True)`;
- bonus catch/effect/expiry support in `vector_runtime.step_many(...)`;

Mostly isolated unless defaults leak:

- seeded bonus arrays unless the wrapper opts in;
- 3P/4P metadata/lifecycle branches;
- scalar/ray `[106]` trainer observations;
- metadata replay artifacts and replay-v0 recorder.

## Current Guardrail Result

After reviewing the current docs/code, the focused source-state visual trainer
checks pass locally, but the no-death profile rerun found an Environment runtime
blocker:

```text
uv run pytest \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py \
  tests/test_vector_visual_observation.py \
  tests/test_curvyzero_stacked_debug_visual_survival_lightzero_smoke.py \
  tests/test_lightzero_phase_profiler.py -q

23 passed
```

`py_compile` also passed for the source-state visual wrapper, renderer,
`VectorMultiplayerEnv`, and Modal trainer script.

Additional focused checks after the current source-state/natural-runtime wiring:

```text
uv run pytest \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py -q

19 passed

uv run pytest tests/test_vector_multiplayer_env.py -q -k "natural_bonus or death_mode"

7 passed, 68 deselected
```

The Modal profile rerun then selected the right path but failed on a naturally
spawned bonus catch:

```text
run_id=opt-source-state-nodeath-profile-c16-sim16-s1161
env_variant=source_state_fixed_opponent
underlying_env_class=VectorMultiplayerEnv
runtime_env_impl_id=curvyzero_vector_multiplayer_metadata_natural_bonus_spawn/v0
death_mode=profile_no_death
error=unsupported caught bonus type code 11
```

Type code `11` is `BonusAllColor`. This catches a real shared-runtime gap that
tests did not cover. The source-state visual route is wired, but the long
profile cannot be treated as valid until Environment resolves the
source-default bonus catch/effect gap.

## Optimizer Read

This is not "training on an irrelevant version." It is training on the current
shared runtime under hardening, through a narrow LightZero adapter.

The risk is real: if Environment changes default 2P no-bonus reset/step/state
semantics, optimizer numbers can move and coach runs can change. The right
response is not to pretend the runtime is full fidelity. The right response is:

1. keep the trainer artifact metadata explicit about `env_variant`,
   `underlying_env_class`, visual schema, and fidelity level;
2. rerun focused source-state wrapper tests after Environment changes;
3. rerun a small native profile when shared reset/step/render code changes;
4. ask Environment to review any claim that promotes the visual tensor beyond
   source-state geometry raster.
5. hand Environment runtime semantic blockers, such as the current
   `BonusAllColor` catch failure, instead of patching source fidelity from the
   optimizer lane.
