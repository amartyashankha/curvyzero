# Environment Reconstruction Strategy Review

Status: working strategy note
Date: 2026-05-09

## Top Summary

The work has been slow for two different reasons.

Some care was necessary. CurvyTron has source rules that are easy to get wrong:
reverse player update order, point insertion before collision, PrintManager
toggles after collision, visual trail clears that do not remove world bodies, and
own-body latency by body number. Those facts affect real gameplay, so pinning
them with small JS/Python canaries was the right move.

Some drag was self-inflicted. We spent too much time proving each narrow slice as
if it were the first slice, let docs repeat status across too many places, kept
side lanes warm in the main thread, and did not set a clear stop rule for source
reading. The result felt careful but heavy.

The faster plan is: build from a single source-spec map, run one narrow claim at a
time through the oracle/Python loop, parallelize independent reads and fixture
design, and only improve runners when the current mismatch truly needs it.

## What Was Necessary

- Source order matters. A tiny before/after mistake changes who dies, which body
  exists, and which score event fires.
- Trails have two meanings: visible trail points and collision world bodies.
  Treating them as one state would have created false gap safety.
- Same-frame behavior is real, not polish. A point can materialize and collide in
  the same source update.
- Small canaries are useful. They made movement, wall death, body latency,
  PrintManager toggles, trail cadence, and first trail-gap behavior concrete.
- Common trace was worth it. Raw JS and Python runner output are different enough
  that comparing raw artifacts would keep wasting time.

## What Slowed Us Down

- The same loop was re-argued too often. Once the local oracle/common-trace/batch
  path worked, new slices should have reused it with less ceremony.
- Status is spread across many docs. The docs are valuable, but several files now
  repeat the same latest checkpoint.
- Some source reads kept going after the fixture claim was already answerable.
  More reading is useful only when it changes the expected trace or prevents a
  likely wrong implementation.
- Runners risk becoming abstractions too early. A narrow runner is fine; a broad
  runner framework before the rules settle is drag.
- Optimization, Modal, browser, and training boundaries stayed visible, but they
  sometimes competed for attention with the current fidelity slice.

## Faster Strategy

Use one full source spec as the map, not many local theories. The spec should say,
in plain language, the source order and state meaning for movement, printing,
body insertion, collision, scoring, round lifecycle, bonuses, and observations.
Existing source-map docs can hold the detail; the active working docs should link
to the current rule and avoid copying it everywhere.

For each new behavior, make one claim and one fixture first. The completed
`source_trail_gap_print_to_hole_boundary_kills_step` proved only that the
important boundary body from `setPrinting(false)` can kill a later player in the
same source update. It did not also become the place to solve broader same-frame
order, exact threshold timing, borderless trails, or observations.

Parallelize safely:

- Source reading for independent mechanics: one reader on PrintManager and
  `setPrinting`, one on collision/body lifetime, one on scoring/events.
- Fixture expectation review: another worker can sanity-check geometry, player
  order, event order, and counters before code changes.
- Artifact triage: after a failing run, one worker reads common traces while
  another checks source facts and another checks the Python runner shape.
- Regression batches: run unrelated promoted batches in parallel when the code
  change is done.
- Docs cleanup: one worker updates coverage and active-lane notes after the proof
  is stable.

Keep sequential:

- The claim definition. The main thread picks exactly one behavior before workers
  split.
- JS oracle before Python behavior changes for that claim.
- Shared runner, normalizer, or schema edits. These touch the comparison surface
  and need one owner at a time.
- Promotion to batch. Only promote after the one-scenario common trace is boring.
- Final status write. One concise checkpoint should update the working memory.

## Runner Rule

Do not build a runner framework because a future slice might need it. Add the
smallest runner support that the current source claim needs.

Good runner changes:

- Add one explicit scenario id.
- Add one trace field needed to explain the current mismatch.
- Reuse existing common-trace projection when possible.
- Keep fixture-specific setup in the scenario when it is clearer than a helper.

Bad runner changes:

- New generic replay layers before several fixtures need the same shape.
- Browser or pixel gates for server gameplay state.
- Broad print/trail/collision abstractions that hide the source order.
- Training-facing APIs that depend on source-runner internals.

Promotion rule: if the same setup or projection code is copied three times and
the source rule is stable, then extract a helper. Before that, clarity wins.

## Source Reading Stop Rule

Source reading is enough when all four are true:

- The source path and update order are named.
- The exact state change needed by the fixture is known.
- The expected common-trace fields and events are listed.
- Any unknown detail is not needed for this claim and is recorded as a later
  probe.

Keep reading or probing when the behavior depends on event order, collision body
lifetime, timer/random state, same-frame death/scoring, client/server split, or a
numeric threshold. If the source is clear and the risk is low, skip a separate
hard-coded probe and go straight to a scenario plus JS oracle.

## Recommended Operating Loop

For each next slice:

1. Name one source claim in one sentence.
2. Read only the source needed to predict that trace.
3. Write the expected event and counter table before Python changes.
4. Run the JS oracle for the scenario.
5. Implement the smallest Python/common-trace change needed to match it.
6. Run the targeted batch, then the neighboring regression batches.
7. Update coverage and active lanes in five lines or less.
8. Stop after validation unless the user explicitly asks to start the next
   independent source read.

Completed gap-loop history:

1. `source_trail_gap_print_to_hole_boundary_kills_step` is in the JS oracle
   fixtures.
2. The JS trace shows p1 boundary point, p1 printing false, p0 death by p1, and
   `worldBodyCount: 2`.
3. The scenario is promoted into `source-trail-gap-canary`.
4. `source_trail_gap_batch.json` later expanded and now covers four forced gap
   fixtures, including the hole-to-print same-update emitted-body case.
5. Do not start same-frame order or head-head cases as part of this cleanup.

## Practical Guardrails

- Keep the main thread about fidelity until trail/collision behavior is boring.
- Keep optimization/vectorization as a shape note, not a coding lane.
- Keep Modal/browser/pixels parked unless state traces cannot answer the question.
- Keep docs short at the top. Put messy evidence in investigation notes.
- Prefer one explicit fixture over one clever abstraction.
