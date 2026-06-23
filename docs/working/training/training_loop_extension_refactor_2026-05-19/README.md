# Training Loop Extension Refactor

Last updated: 2026-05-19

This folder is its own task lane. Do not use the higher-level Curvytron feedback-loop docs as the scratchpad for this refactor. Keep planning, orchestration, findings, and task state here unless a short pointer from another doc is truly needed.

## Goal

Make the training code easier to change when we want algorithmic changes, such as:

- changing the opponent batch mixture,
- enforcing a consistent policy observation perspective,
- changing reward contracts,
- adding an exploration bonus,
- eventually training a side network alongside the main policy.

The tournament and multi-run control plane are mostly out of scope for this lane. We only care about the interfaces where the trainer reads opponents/checkpoints and publishes checkpoints.

## Current Working Truth

- The stock LightZero hookup is small, but it is buried inside a large Modal trainer harness.
- Reward math, LightZero support bounds, env telemetry, config patching, checkpoint metadata, and resume hooks are too coupled.
- Future algorithm changes should not require editing one giant Modal file plus the env plus several launch manifests by hand.
- This refactor should first make boundaries explicit, then make behavior-preserving extractions, then add new behavior behind clear contracts.

## Documents

- `CURRENT_PHASE.md`: current state, decisions, active lanes, and gates.
- `ORCHESTRATION.md`: main-thread/subagent coordination for this task only.
- `OPERATING_PATTERNS.md`: how to operate on this refactor lane without losing state.
- `TASK_BOARD.md`: concrete work items and done criteria.
- `COMPATIBILITY_LEDGER.md`: temporary facades, private imports, and fallbacks that are allowed only with expiry/tests.
- `FINDINGS_LOG.md`: append-only findings from subagents and main-thread investigation.
- `SOURCE_MAP.md`: lightweight index of code hotspots for this lane.
- `DECISION_LOG.md`: accepted decisions and why they were made.
- `GLOSSARY.md`: plain-language definitions for overloaded training terms.
- `TRAINER_BLOAT_CRITIQUE.md`: direct critique of why the Modal trainer became too large.
- `REWARD_CONTRACTS_PLAN.md`: reward/support extraction plan.
- `LIGHTZERO_CONFIG_BUILDER_PLAN.md`: config-builder extraction plan.
- `EXPERIMENT_KNOB_SURFACE.md`: compact experiment-facing knobs versus normalized internal config.
- `COMPACT_MANIFEST_ROW_SCHEMA.md`: target grouped-submit row shape and compact-row migration rule.
- `HOOK_BUNDLE_AND_EXTENSION_PLAN.md`: LightZero hooks, resume, and extension interface plan.
- `ENV_STEP_MODULARITY_PLAN.md`: env reset/step/reward/observation boundary plan.
- `BATCH_CONSTRUCTION_PLAN.md`: opponent slot and batch-split investigation/design plan.
- `EXPERIMENT_BATCH_INVENTORY.md`: side-lane inventory of recent local experiment batches and which knobs were fixed or varied.
- `TEST_AND_MIGRATION_PLAN.md`: tests and migration gates.

## Rules For This Lane

- Keep docs current before and after meaningful changes.
- Use subagents for bounded read-only audits first; main thread integrates the conclusions.
- Do not launch Modal jobs as part of this refactor unless the task board explicitly changes.
- Do not preserve obsolete defaults just for compatibility. Make the intended defaults clear and test them.
- Do not do giant rewrites. Extract one boundary at a time with tests around the behavior that matters.

## Hard Gates

- Trainer patch gate: any edit to the Modal trainer must declare one outcome first: pure extraction, side-effect wrapper, temporary compatibility shim, or documented exception.
- Temporary compatibility shims must name the protected run or caller, expiry/deletion criteria, and test coverage.
- Local validation before launch: Modal runs are only for a named remote/runtime question after local reward/config/hook gates pass.
- Subagent integration gate: do not start another broad scout batch until prior findings have been reduced into findings, decisions, and task-board changes.
- No hidden fallback: unlisted fallback/default behavior in a touched path should be deleted or explicitly entered into the compatibility ledger.
