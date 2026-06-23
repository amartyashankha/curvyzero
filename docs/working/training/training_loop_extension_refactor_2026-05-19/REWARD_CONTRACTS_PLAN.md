# Reward Contracts Plan

Last updated: 2026-05-19

## Why This Exists

Reward behavior is central to training, LightZero support bounds, telemetry, and future exploration-bonus work. It should not be split across env code, Modal config patches, and manifests.

## Current Locations

- `src/curvyzero/training/reward_contracts.py` now owns the shared reward/support contract.
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py` still computes scalar rewards at runtime, but delegates reward schema/hash/perspective/space metadata to the shared contract.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py` keeps facade helpers for existing trainer callers, but those helpers delegate to the shared contract.
- `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py` imports reward normalization/support config from the shared contract.

## Current Shape

`src/curvyzero/training/reward_contracts.py` owns:

- reward variant names,
- reward schema ids and hashes,
- reward component metadata/policy dictionaries,
- reward-space bounds,
- LightZero support-size calculation,
- `auto` reward normalization by env variant,
- validation of unsupported reward settings.

## Completed In This Cut

- Added focused public tests in `tests/test_reward_contracts.py`.
- Wired source-state env metadata helpers to the shared module.
- Wired trainer reward/support helpers to delegate to the shared module.
- Wired eval model-target support config to the shared module.
- Kept trainer reward constants as imports from the shared module so current callers still have a stable facade without duplicate definitions.
- Accepted that `auto` belongs in the public reward contract for trainer/config inputs, while env-facing configs receive normalized concrete variants.
- Left scalar reward math in the env. The contract owns metadata/support/bounds; runtime reward calculation has not moved.
- Documented current `policy_action_repeat_max` behavior through reward-space bounds; LightZero support range is unchanged for now.

## Remaining Work

- As config-builder extraction lands, move tests and non-entrypoint imports away from trainer-private helpers.
- Keep runtime scalar reward math in the env until a smaller env reward-component helper is worth extracting.
- Do not add intrinsic reward here until the contract has a named variant, bounds, support calculation, and focused tests.
- Decide later whether canonical eval/status reward components should expose `sparse_outcome` and `terminal_outcome` for every variant.

## Done Criteria

- Reward contract can be imported by env/config code without import cycles.
- Existing reward behavior is unchanged.
- Invalid reward settings fail loudly.
- Future intrinsic reward work has one clear place to add new reward components.
