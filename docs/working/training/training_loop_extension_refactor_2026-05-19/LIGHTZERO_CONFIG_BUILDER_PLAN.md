# LightZero Config Builder Plan

Last updated: 2026-05-19

## Why This Exists

The trainer should not hide the core LightZero config contract inside a large Modal file. Config construction should be small enough to test locally and reason about.

## Current Locations

- `src/curvyzero/training/lightzero_config_builder.py` now owns path patching, LightZero target support patches, checkpoint hook config patches, env-variant specs, render/backend/seat/cadence validators, opponent relation helpers, visual-survival surface extraction, and visual-survival config construction.
- The primary local boundary is `VisualSurvivalConfigSpec -> build_visual_survival_config(...) -> VisualSurvivalConfigResult`.
- The compact experiment-facing boundary is `VisualSurvivalExperimentSpec -> visual_survival_config_spec_from_experiment(...) -> VisualSurvivalConfigSpec`.
- `VisualSurvivalConfigSpec` is grouped normalized config, not the user-facing experiment surface. It separates run/runtime, training scale, timing, observation, behavior, reward/target, and opponent config so low-level LightZero/env fields do not look like blessed launch knobs.
- The broad public `build_visual_survival_configs(**kwargs)` remains as a same-signature facade for trainer/eval compatibility and delegates through `VisualSurvivalConfigSpec.from_builder_kwargs(...)`.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py` keeps `_build_visual_survival_configs(...)` as a facade that delegates to the public builder.
- Eval imports the public builder instead of trainer-private `_build_visual_survival_configs(...)`.
- defaults in `src/curvyzero/contracts/curvytron.py`
- launch manifests and CLI arguments that patch config values

## Target Shape

Keep `src/curvyzero/training/lightzero_config_builder.py` centered on one typed builder:

- input: `VisualSurvivalConfigSpec`, including nested `FrozenOpponentConfig`;
- grouped inputs: `VisualSurvivalRunRuntimeSpec`, `VisualSurvivalTrainingScaleSpec`, `VisualSurvivalTimingSpec`, `VisualSurvivalObservationSpec`, `VisualSurvivalBehaviorSpec`, `VisualSurvivalRewardTargetSpec`, `VisualSurvivalOpponentSpec`;
- output: `VisualSurvivalConfigResult`;
- result fields: `template_module`, `main_config`, `create_config`, `surface`, `patches`;
- result properties: `env_config`, `lightzero_target_config`;
- validation continues to raise rather than being stored in the result.

## Initial Tasks

- Keep current helper/builder tests passing without importing Modal.
- Add or preserve builder gates for fixed hard-coded, frozen checkpoint, opponent mixture, assignment context, learner-seat mode, and reward support.
- Keep the signature-drift test proving the broad facade and typed spec agree on every keyword.
- Keep parity tests for typed result vs broad facade.
- Keep the unknown-kwarg rejection test so stale settings fail instead of silently flowing through.
- Keep compact experiment-surface tests proving current broad defaults expand correctly and internal launch knobs are rejected.
- Delete trainer-private helper wrappers once all local callers import public modules directly.

## Completed In This Cut

- Added `src/curvyzero/training/lightzero_config_builder.py`.
- Added `tests/test_lightzero_config_builder.py`.
- Moved `target_config_patches`, `set_save_ckpt_after_iter`, `set_load_ckpt_before_run`, `set_or_add_path`, `get_path`, `to_plain`, `env_variant_spec`, validators, opponent relation helpers, `build_visual_survival_configs`, and `extract_visual_survival_surface` behavior behind the public module.
- Updated eval to import public `build_visual_survival_configs` and `target_config_patches`.
- Added `VisualSurvivalConfigSpec`, `FrozenOpponentConfig`, `VisualSurvivalConfigResult`, and `build_visual_survival_config(...)`.
- Grouped `VisualSurvivalConfigSpec` internally so the builder distinguishes normalized config groups from experiment-facing knobs.
- Added `VisualSurvivalExperimentSpec` and `build_visual_survival_config_from_experiment(...)` as the compact target surface for future manifest/grouped-submit cleanup. Its field list is pinned so internal LightZero/env knobs do not leak back into the experiment surface.
- Made the broad same-signature builder a facade through the typed spec/result path.
- Kept trainer wrappers as ledgered facades while launch/tests continue to use the trainer module as an entrypoint.

## Done Criteria

- Config can be built in a unit test without Modal.
- The Modal trainer delegates to this builder.
- Defaults are explicit and current.
- Unsupported old settings do not silently survive.
- Eval imports no trainer-private config helpers.
- The broad facade remains only while protected callers still need it and is listed in the compatibility ledger.
