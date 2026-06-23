# Training Loop Modularity Plan

Created: 2026-05-19.

Purpose: identify the small set of training-code boundaries that matter for
future algorithm changes, such as an exploration bonus or a trainable side
network. This is not a plan to clean up every large file.

## Plain Diagnosis

The tournament/multiple-run setup is not the main problem here. It can stay
as an experiment control plane.

The training-code problem is that algorithm ideas are not isolated. A change
like "add an exploration bonus" currently touches too many places:

- CLI/default arguments in the Modal trainer;
- LightZero config patching;
- env reward calculation;
- reward support-size calculation;
- checkpoint metadata;
- resume/checkpoint sidecar state;
- telemetry and summary fields;
- tests and manifest builders.

That makes a real training-loop change feel like editing the whole system.

## Current Important Files

| File | Current Role | Problem |
| --- | --- | --- |
| `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py` | Builds LightZero config, installs hooks, writes artifacts, spawns eval/GIF work, owns CLI. | Too many responsibilities. It should mostly orchestrate. |
| `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py` | Custom LightZero env: reset, step, reward, observation, opponent action, telemetry. | Reward, observation, opponent selection, and telemetry are coupled. |
| `src/curvyzero/contracts/curvytron.py` | Shared current defaults and Modal object names. | Useful, but experiment knobs and algorithm knobs are mixed. |
| `src/curvyzero/training/opponent_mixture.py` | Opponent-slot parsing and deterministic collector-env split. | This is already closer to the right shape. |
| `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py` | Loads frozen checkpoint opponents and enforces observation metadata. | Also reasonably bounded. |

## The Boundary We Need

Introduce a training-extension layer. The extension should be the one place
where a new algorithm feature declares:

- config fields and defaults;
- env reward/state hooks;
- LightZero config patches;
- learner/replay hooks, if needed;
- checkpoint/resume sidecar state;
- telemetry fields;
- validation/proof checks.

The Modal trainer should only receive a resolved `TrainingRunSpec`, install the
declared extension hooks, call LightZero, and write artifacts.

## Proposed Modules

### `training/lightzero_config_builder.py`

Move `_build_visual_survival_configs`, `_extract_surface`, and
`_validate_visual_survival_surface` out of the Modal file.

This module should accept a typed `TrainingRunSpec` and return:

- `main_config`;
- `create_config`;
- `surface`;
- `patches`;
- `env_config`;
- validation problems.

Why this matters: changing LightZero knobs should not require touching Modal
launch code.

### `training/reward_contracts.py`

Move reward variant definitions, reward support-size calculation, and reward
schema metadata out of both the Modal trainer and the env file.

This module should define:

- `RewardSpec`;
- reward component names;
- reward-space bounds;
- LightZero support config;
- telemetry labels.

Why this matters: exploration bonuses and reward shaping need a single source
of truth for "what reward did the learner train on?"

### `training/env_step_modules.py`

Split the source-state env step into composable pieces:

- learner-seat selector;
- opponent actor;
- reward computer;
- observation renderer;
- telemetry builder.

Do not rewrite the env. Start by extracting pure helpers used by the current
env.

Why this matters: the env is the natural place for non-trainable reward shaping
and state-derived intrinsic rewards. It should not require editing a giant
`step()` method.

### `training/lightzero_hooks.py`

Move hook installers out of the Modal file:

- checkpoint progress writer;
- live checkpoint publisher;
- resume sidecar hooks;
- learner metrics recorder;
- target audit;
- assignment refresh hook.

Expose them through one `HookBundle`:

```text
bundle = build_hook_bundle(spec, artifacts, extensions)
restore = bundle.install(train_muzero)
try:
    train_muzero(...)
finally:
    restore()
```

Why this matters: a side network will need learner hooks and checkpoint/resume
state. It should be one extension, not another pile of local `restore_*`
variables.

### `training/extensions/`

Add explicit extension modules. The first useful shape:

```text
TrainingExtension
  id
  config_schema
  build_env_config_patch()
  build_lightzero_config_patch()
  install_learner_hooks()
  checkpoint_state()
  restore_checkpoint_state()
  telemetry_fields()
  validate_surface()
```

This does not need to be fancy. It just needs to make the boundary explicit.

## Exploration Bonus: What Kind?

There are two very different ideas hidden under "exploration bonus."

### 1. Env-Side Bonus

Example: reward novelty for visiting a new region, distance from old trail,
or hitting rare board states.

This can be implemented inside the env/reward layer. It is the simplest path.

Pros:

- LightZero still sees a normal scalar reward.
- No custom learner required.
- Checkpoint format barely changes.

Cons:

- The bonus is hand-coded or env-local.
- It cannot use a learner-trained side network unless that network is somehow
  made available to collectors.

Recommended first path if we want speed.

### 2. Trainable Side Network

Example: curiosity model, random-network-distillation predictor, density model,
or auxiliary value head.

This is a learner extension, not just a reward variant.

Extra needs:

- side network definition;
- side optimizer;
- learner hook that sees replay batches;
- checkpoint/resume sidecar for side network + optimizer;
- telemetry for side loss and intrinsic reward scale;
- clear decision on whether intrinsic reward is computed during collection or
  only as an auxiliary training loss.

Hard part:

- If intrinsic reward affects collection rewards, collectors need access to the
  current side network. Stock LightZero does not make that easy.
- If the side network only adds an auxiliary loss, it is easier, but it does
  not directly change the reward targets already stored in replay.

Recommended first trainable-side-network path:

1. Add it as an auxiliary learner loss/telemetry only.
2. Prove side network trains and checkpoint/resume works.
3. Only later decide whether to feed intrinsic reward back into collection.

## Minimal Refactor Sequence

Do only the parts that unlock training-loop changes:

1. Extract reward contracts.
   - Move reward specs and support-size calculation into
     `training/reward_contracts.py`.
   - Keep behavior byte-for-byte equivalent where possible.
   - Add focused tests for reward bounds and support config.

2. Extract LightZero config builder.
   - Move `_build_visual_survival_configs`, `_extract_surface`, and validation.
   - Modal trainer still calls the same function, now imported.
   - Add one config-builder test that proves current broad defaults compile.

3. Extract hook bundle.
   - Do not change hook behavior.
   - Replace local `restore_*` sprawl with one install/restore object.
   - This is the first place a side network should plug in.

4. Add extension interface.
   - Start with a no-op extension and a reward-only extension.
   - Do not implement a curiosity network until the interface can carry config,
     telemetry, and checkpoint sidecar state.

5. Then implement one small exploration bonus.
   - Prefer env-side novelty first.
   - Use it to prove the extension boundary before adding a trainable network.

## What Not To Do Yet

- Do not refactor the tournament for this task.
- Do not rewrite the whole env class.
- Do not replace LightZero before we have a cleaner extension boundary.
- Do not add a trainable side network by directly monkey-patching random code in
  the Modal trainer.
- Do not add more bare CLI flags unless they map to a typed spec or extension
  config.

## Best Next Concrete Change

The highest-leverage first patch is:

```text
extract reward contracts + LightZero target support config
```

Reason: any exploration bonus has to answer two questions immediately:

- What scalar reward does LightZero train against?
- What support range/scale does the MuZero reward/value head use?

Those answers are currently split between the Modal trainer and the env. They
should be one module before adding more reward complexity.

The second patch should extract the LightZero config builder. Once those two
are done, new training-loop variants become much less invasive.
