# Tournament Refactor Bridge

Last updated: 2026-05-16.

## Why This Exists

The tournament code grew too large while we were proving the feedback loop. This
doc connects older refactor notes to the current research phase so cleanup does
not fight live-learning validation.

## Current Risk

The large files are hard to reason about:

- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`

The danger is not just line count. The danger is mixed responsibilities:

- Modal app setup
- CLI modes
- Volume paths
- browser/web endpoints
- checkpoint intake
- scheduler policy
- game spawning
- rating reduction
- artifact export
- cleanup tools

## Cleanup Rule

Refactors should be behavior-preserving and staged.

Good cuts:

- pure scheduler helpers;
- rating math;
- artifact path helpers;
- website read models;
- Modal settings/defaults;
- browser render helpers;
- command wrappers around stable helpers.

Bad cuts:

- broad rewrites during active validation;
- changing scheduler behavior while moving files;
- changing artifact paths without migration;
- splitting code before tests identify the contract.

## Allowed During Scheduler Research

- Extract pure local simulation helpers.
- Add docs and tests that describe current behavior.
- Add read-only probes.
- Extract small pure scheduler/rating helpers if behavior is locked by tests.
- Add scheduler observability fields if they do not change scheduling behavior.

## Deferred Until Scheduler Choice

- Changing artifact paths.
- Moving Modal orchestration entrypoints.
- Replacing `adaptive_v0`.
- Changing public leaderboard semantics.
- Changing trainer-facing assignment selection.
- Large file splits that mix behavior changes with movement.

## Current Bridge

Use these current docs first:

- `CURRENT_RESEARCH_PHASE.md`
- `SCHEDULER_SIMULATION_PLAN.md`
- `ORCHESTRATION.md`
- `TOURNAMENT_ARCHITECTURE_CRITIQUE.md`
- `OBSERVABILITY_CONTRACT.md`
- `POLICY_OBSERVATION_CONTRACT.md`

Older checkpoint tournament docs remain evidence, not command authority, unless
they are linked from the current docs.
