# Hook Bundle And Extension Plan

Last updated: 2026-05-19

## Why This Exists

Checkpoint publishing, progress writing, resume state, opponent refresh, and future side-network training all depend on hooks around LightZero. Those hooks need a clear boundary.

## Known Hook Locations

- `_install_live_checkpoint_publisher`: `lightzero_curvyzero_stacked_debug_visual_survival_train.py`, line 1781.
- `_install_checkpoint_progress_writer`: `lightzero_curvyzero_stacked_debug_visual_survival_train.py`, line 2355.
- `_install_lightzero_full_resume_state_hooks`: `lightzero_curvyzero_stacked_debug_visual_survival_train.py`, line 2480.
- `_install_lightzero_opponent_assignment_refresh_hook`: `lightzero_curvyzero_stacked_debug_visual_survival_train.py`, line 6057.
- learner-metric and target-audit hook code in the Modal trainer

## Target Shape

Create a hook module, likely `src/curvyzero/training/lightzero_hooks.py`, with:

- a `LightZeroSymbols` resolver for LightZero classes/functions,
- a `PatchRegistry` that owns all patch/restore order,
- a `HookBundle` object,
- explicit hook install order,
- checkpoint metadata writers,
- resume-state readers/writers,
- opponent-assignment refresh behavior,
- optional extension hook registration.

For future research, add a small `TrainingExtension` interface only after the hook boundary is understood.

## Side-Network Questions

- Where can a side model see learner batches?
- Where can it store optimizer state?
- How is it checkpointed and resumed?
- Can collectors use it, or is it learner-only at first?
- Does it change policy rewards, learner loss, or only telemetry?

## Done Criteria

- Existing hooks are installed from one clear place.
- Hook order is explicit.
- Checkpoint/resume side effects are documented and testable.
- A no-op extension can be registered without changing training behavior.

## Current Extraction Gate

Do not start hook extraction until pure reward, opponent, config, metadata, and progress/status contracts are stable.

First hook extraction when ready:

- add a stacked-order test for checkpoint save wrappers;
- extract `PatchRegistry`;
- extract only the checkpoint-save bundle around progress writer and live publisher;
- leave resume, opponent refresh, and initial-policy loading in place for a later pass.
