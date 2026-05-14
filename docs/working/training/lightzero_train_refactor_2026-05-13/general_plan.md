# General Plan

## Phase 0: Grounding

Create this planning directory, record the active facts, and delegate bounded
audits. Keep source files unchanged except for later test and bugfix patches.

## Phase 1: Regression Test Lockdown

Add focused tests for the behavior that must survive the refactor:

- checkpoint discovery across `lightzero_exp*`;
- latest-checkpoint selection;
- progress/status latest checkpoint reporting;
- auto-resume checkpoint and sidecar selection;
- checkpoint poller candidate discovery;
- background eval/GIF metadata paths;
- manifest/tournament inputs that freeze checkpoint refs.

These tests should use local temp directories and stubs where possible. Modal
itself should not be required for the core contract tests.

## Phase 2: Minimal Bugfix In Place

Fix the checkpoint-discovery bug in the current file layout first. The fix
should make all relevant readers use one broad discovery helper. Keep the patch
boring.

## Phase 3: Extract Pure Helpers

Only after tests pass, extract helpers in small steps:

1. checkpoint discovery and checkpoint refs;
2. resume and sidecar selection;
3. progress/status payload construction;
4. poller checkpoint queue construction;
5. eval/GIF configuration payloads;
6. Modal-only wrappers.

Each extraction keeps the old public entrypoint stable until tests say the move
is safe.

## Phase 4: Clean Naming And Taxonomy

Replace repeated strings with named constants only where they cross module,
artifact, or website boundaries. Do not create broad abstraction layers for
one-off local values.

## Phase 5: Retire Stale Tests And Docs

After the refactor lands and focused tests cover the new contracts, delete or
archive tests and docs that only describe dead paths. Do not delete evidence
needed to understand past run results.

