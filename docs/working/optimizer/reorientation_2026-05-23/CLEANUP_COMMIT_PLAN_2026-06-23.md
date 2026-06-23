# Cleanup / Commit Plan

Date: 2026-06-23

This is a cleanup snapshot, not a new goal file. `goal.md` remains the durable
compass.

## Current Believable State

- Accepted baseline is still OPT-104:
  `12689.38 env/s`, `14.5255s` wall, H100, B1024/A1, normal death, same-work
  full training loop.
- The fastest single same-window support row is the corrected columnar
  append/direct-table owner-search stack:
  `15852.67 env/s`, `46.7666s`, about `1.25x` OPT-104.
- The best repeat/longer evidence is weaker than that single row. The
  threaded/background owner-maintenance candidate repeated above OPT-104 and
  its longer row was positive but modest: `13145.34 env/s`, about `1.036x`.
- Nothing is promotion-grade yet. The near target is still a repeatable `2x`
  same-work H100 win.
- Replay/sample/layout-only work is no longer a credible standalone path to
  `2x`. Conservative replay/sample ceiling on the fastest row is about
  `24903.25 env/s`, or `1.9625x`, still short of `2x` before accounting for
  risk and variance.

## What Is Actually Optimized

- Vectorized reset/autoreset RNG with exact scalar parity is a real accepted
  speed patch and must be preserved.
- Refresh interval `4` is the accepted cadence for same-work comparison.
- Threaded/background owner maintenance is the preserved candidate shape.
- Owner-search action-only/replay ownership is real:
  parent committed/stored replay rows are `0`, search result payload bytes are
  `0`, and the owner materializes replay.
- Ring-batched replay append/cache refresh, direct record-table build, and
  columnar append are useful support layers.
- Direct-root request, resident-root view, host-observation stub, owner-proxy
  transition closure, fixed action-result slot, action-dispatch overlap,
  owner mechanics step-view, dense owner action publication, and fixed-SoA/
  owner-slot proofs are support gates. They are not promoted speed lanes by
  themselves.

## Algorithm In Plain Terms

The algorithm is still MuZero-style compact self-play:

- vectorized CurvyTron environments produce observations/root state,
- compact Torch search runs batched one-step/root action search,
- selected actions are applied to the next mechanics step,
- compact replay stores transition/search-derived training data,
- learner samples unroll-2 batches and updates the model,
- owner/search/replay paths increasingly exchange handles, slots, generations,
  and digests instead of parent Python materializing full objects.

The architecture target is Puffer/EnvPool/Isaac-like ownership: mechanics,
search, replay, sample, and learner own fixed resident buffers; parent Python
coordinates epochs, proof, launch, and final drain.

## Current Main Lane

Build the next owner replay/sample/learner handle rung, not another unchanged
H100 row:

- production sample returns an owner-issued resident batch/window handle,
- learner consumes that handle or reports explicit materialized-parent
  fallback,
- parent replay/sample objects and selected-group loops stay zero where
  claimed,
- normal-death/action/terminal/tensor-native proof remains closed,
- local corrected whole-loop timing shows a real moved owner surface before an
  H100 launch.

## Cleanup Done In This Pass

- Removed the half-wired post-submit owner-proxy transition-closure experiment.
  It had no direct-stepper flag, no tests, and would have introduced a
  one-request replay lag without proof. The live code is back to the validated
  pre-submit owner-proxy closure contract.
- Validation after cleanup:
  - `uv run ruff check ...` on the scoped owner-search/speed-row/source-state
    files passed.
  - Focused owner-proxy closure tests passed: `3 passed`.
  - Recent speed-row expected-train and owner-manager boundary regressions
    passed: `2 passed`.

## Commit Recommendation

Do not make a blind broad commit from the current tree.

Reasons:

- The branch is `main` at `af24299 Save final full loop note`.
- There are many existing tracked modifications outside the compact optimizer
  lane.
- The compact optimizer stack is largely untracked relative to `HEAD`, so a
  self-contained commit must include its dependencies deliberately, not just
  the files touched in the latest pass.

Recommended commit series:

1. `optimizer-docs-reorientation`: `goal.md` plus
   `docs/working/optimizer/reorientation_2026-05-23/`.
2. `compact-core-contracts`: compact policy/search/replay/observation/learner
   support modules and their focused tests.
3. `compact-owner-search-stack`: owner-search service, rollout slab, speed-row
   smoke/report plumbing, and owner-search tests.
4. `hybrid-owner-boundaries`: source-state hybrid owner-boundary code and
   tests.
5. Separate legacy/training/tournament cleanup commits only after reviewing
   the older tracked dirty files.

Each commit should be staged by explicit path list and validated with focused
tests before committing.
