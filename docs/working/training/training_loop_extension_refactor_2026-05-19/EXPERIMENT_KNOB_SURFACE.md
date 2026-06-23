# Experiment Knob Surface

Last updated: 2026-05-19

This doc separates the small experiment-facing surface from the normalized internal LightZero/env config.

## Plain Rule

Do not expose every internal builder field as a launch knob. The builder needs many fields because LightZero and the source-state env need fully normalized config. Experiment manifests should name the few things we intentionally vary.

## Experiment-Facing Knobs

These are reasonable top-level experiment choices in manifest/orchestration space:

- run identity: `seed`, `run_id`, `attempt_id`;
- scale preset: current broad lane, canary/proof lane, or explicit ablation;
- reward family and `reward_outcome_alpha`;
- opponent source: assignment ref, inline mixture spec, frozen checkpoint, or hard-coded policy;
- leaderboard-opponent immortality policy, when it is intentionally part of the recipe;
- action noise level;
- learner seat mode, when we intentionally ablate it;
- initial policy checkpoint;
- checkpoint cadence and refresh cadence, when canary/proof runs intentionally shorten them.

Only the subset needed by local config construction belongs in `VisualSurvivalExperimentSpec`; launch-only choices such as run id, attempt id, initial checkpoint, and checkpoint/refresh cadence should live in manifest/runtime normalization rather than the LightZero config-builder spec.

## Normalized Internal Config

These should stay internal unless a test explicitly opens an ablation:

- `env_variant=source_state_fixed_opponent`;
- `decision_ms`, source physics step, source-step semantics;
- current policy trail/bonus render modes;
- `policy_observation_backend`;
- `lightzero_multi_gpu`;
- profile flags;
- telemetry stride;
- default target support cap and `td_steps`;
- background eval/GIF defaults;
- null checkpoint/snapshot placeholder fields;
- individual LightZero template patch paths.

## Current Code Boundary

`VisualSurvivalExperimentSpec` is the compact experiment-facing input in `src/curvyzero/training/lightzero_config_builder.py`.

Current fields are intentionally small:

- `seed`;
- `exp_name`;
- `telemetry_path`;
- `reward_variant`;
- `reward_outcome_alpha`;
- `opponent_policy_kind`;
- `frozen_opponent`;
- `opponent_mixture`;
- `opponent_assignment_context`;
- `action_noise_probability`;
- `scale_preset`.

It must not accept `collector_env_num`, `batch_size`, render modes, policy-observation backend, source cadence, `td_steps`, model support caps, or Modal/runtime profile knobs.

It expands to grouped normalized `VisualSurvivalConfigSpec`:

- run/runtime;
- training scale;
- timing;
- observation;
- behavior;
- reward/target;
- opponent.

`build_visual_survival_configs(**kwargs)` remains a compatibility facade for old trainer/eval callers. It is not the design target.

`TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT` is now only the identity/minimum remote-call shell: `mode`, `seed`, `run_id`, and `attempt_id`. The deployed trainer still supplies current defaults for the broader flat Modal function signature.

The submitter now normalizes compact rows before side effects:

- minimal `row["train_kwargs"]` gets the matching poller run identity before spawn;
- optional `row["experiment_spec"]` can expand reward, opponent kind, current scale, and semantic action noise into flat trainer kwargs;
- train-only fields are still rejected from `poller_kwargs`.

`build_curvytron_tonight18_manifest.py` is the compact proof case: it omits default-equal trainer fields from `row["train_kwargs"]`, labels rows with `train_kwargs_schema_id=curvyzero_tonight18_compact_train_kwargs/v0`, and keeps only the row semantics and non-default runtime overrides explicit.

Some manifest builders still emit legacy full `train_kwargs`. That remains the active artifact shape for rows whose values intentionally differ from current trainer defaults. It is not the conceptual required surface.

## Current Scale Preset

`current_broad` expands to:

- `collector_env_num=256`;
- `n_episode=256`;
- `batch_size=64`;
- `num_simulations=8`;
- `max_env_step=30000000`;
- `max_train_iter=300000`;
- checkpoint every 10000 iterations;
- source max steps 1048576;
- current policy observation surface.

## Migration Rule

When touching manifest builders or grouped submitters, prefer accepting compact experiment fields and normalizing before side effects. Do not add another required flat field to `TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT` unless the remote function genuinely cannot run without it.

Do not blindly compact every builder row. First prove the omitted value equals the deployed trainer default. `tonight18` now does this with a local expansion table and validator. `survivaldiag` and `opponent_mixture` still intentionally set non-default collector sizes, checkpoint cadence, background settings, or opponent sources, and those must remain explicit until represented in a compact override schema.
