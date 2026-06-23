# Source Map

Last updated: 2026-05-19

This is a lightweight index, not a full audit. Subagent findings should refine it.

## Modal Trainer Hotspots

File: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`

- `_install_live_checkpoint_publisher`: line 1733.
- `_install_checkpoint_progress_writer`: line 2307.
- `_install_lightzero_full_resume_state_hooks`: line 2432.
- `_run_visual_survival_train`: line 3890.
- stock LightZero call to `train_muzero`: line 4854.
- `_opponent_assignment_refresh_ready_report`: line 5856.
- `_install_lightzero_opponent_assignment_refresh_hook`: line 6012.
- `_build_visual_survival_configs`: facade delegates to public `lightzero_config_builder.build_visual_survival_configs`.
- `_target_config_patches`: facade delegates to public `lightzero_config_builder.target_config_patches`.

Main observation: this file still owns too many side-effect concerns: hook monkey-patching, checkpoint/status side effects, eval/GIF side tasks, CLI defaults, and the final stock LightZero call. Config construction has moved to a public training module.

## Source-State Env Hotspots

File: `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`

- reward/perspective contract imports: near the top-level imports.
- main local env class: line 331.
- reset: line 688.
- step: line 822.
- reward components: line 1581.
- reward schema/hash/perspective helpers: now delegate to `curvyzero.training.reward_contracts`.
- reward space: now delegates to `curvyzero.training.reward_contracts`.

Main observation: the env is the right runtime boundary for LightZero, but its internals need clearer sub-boundaries for perspective, reward, opponent execution, observation, and telemetry.

## Shared Defaults And Contracts

File: `src/curvyzero/contracts/curvytron.py`

- learner seat default is `LEARNER_SEAT_MODE_RANDOM_PER_EPISODE`.
- reward variant names and `CURVYTRON_MAIN_REWARD_VARIANTS` live here.
- grouped submit required kwargs are now only `mode`, `seed`, `run_id`, and `attempt_id`.
- submitter validation still consumes normalized flat trainer kwargs for checkpoint/opponent/policy safety. Legacy full `row["train_kwargs"]` remains accepted for non-migrated builders, while compact rows are normalized before side effects.
- `build_curvytron_tonight18_manifest.py` is the current compact-by-default manifest proof case; it expands compact rows locally for validation and keeps only row semantics/non-default overrides explicit.

Main observation: this file is useful as a shared source of truth, but it mixes algorithm defaults, orchestration knobs, and artifact knobs.

## LightZero Config Builder

File: `src/curvyzero/training/lightzero_config_builder.py`

- `VisualSurvivalConfigSpec` and `FrozenOpponentConfig`: typed builder inputs.
- `build_visual_survival_config`: primary typed builder API.
- `VisualSurvivalConfigResult`: typed builder output.
- `build_visual_survival_configs`: broad compatibility facade returning the historical dict shape.
- `env_variant_spec`, `set_or_add_path`, `target_config_patches`, `extract_visual_survival_surface`: public config helpers.

Main observation: this module now owns the typed visual-survival builder and the pure config helper/spec functions. The broad signature remains only as a compatibility facade.

## Opponent Mixture

File: `src/curvyzero/training/opponent_mixture.py`

- mixture selection unit is `episode_reset`.
- entries use `opponent_immortal` as the explicit boolean; `opponent_death_mode` is derived runtime metadata.
- blank-canvas noop entries must use `fixed_straight` and `opponent_immortal=true`.
- `deterministic_collector_env_mixture_plan` exists and works at `collector_env` granularity, validating slot counts as a power of two, not greater than `env_num`, and dividing `env_num`.

Main observation: some deterministic split machinery already exists. Trainer assignment-refresh plumbing can use this plan when refresh is enabled; the remaining question is how to expose/extract that contract cleanly and which manifests should enable it.

## Immediate Implications

- Reward contracts are the first partial extraction and are wired.
- Config builder extraction and typed spec/result cleanup have landed; the next config step is shrinking protected broad-facade callers, not adding trainer-private helpers.
- Batch construction should not be changed as learner-batch semantics. The known deterministic split is collector-env assignment, while learner `batch_size` is replay sampling.
