# Operator Runbook

## Main Thread Responsibilities

The main thread owns:

- final launch decisions;
- conflict resolution between trainer, tournament, leaderboard, and optimizer lanes;
- source-of-truth docs;
- final preflight checks;
- user-facing summaries.

The main thread should not:

- bury decisions inside subagent outputs;
- launch training from stale manifests;
- treat canaries as policy-strength evidence;
- mix optimizer speed changes with training semantics changes without explicit review.

## Standard Loop

1. Read `README.md`, `current_state.md`, and `overnight_run_decision.md`.
2. Check the latest tournament/training docs for changed evidence.
3. Delegate bounded audits in parallel.
4. Fold returned evidence into the relevant doc.
5. Run the launch-readiness checklist.
6. Ask for explicit approval before launching or changing code.

## Evidence Hygiene

Every claim should identify one of:

- survival eval evidence;
- leaderboard/head-to-head evidence;
- health/liveness evidence;
- optimizer/throughput evidence;
- design-only speculation.

Do not present design-only speculation as implemented system state.

## Subagent Use

Use parallel subagents for:

- trainer assignment contract audit;
- tournament/intake status;
- leaderboard artifact extraction;
- seeded roster/scripted policy design;
- optimizer/speed recommendation check;
- docs critique.

Each subagent should return:

- files read;
- implemented vs designed;
- missing tests;
- specific next gate;
- caveats.

## After Any Evidence Change

Update in this order:

1. narrow evidence doc;
2. `current_state.md` if implementation state changed;
3. `overnight_run_decision.md` if recommendations changed;
4. `launch_readiness_checklists.md` if preconditions changed;
5. final user summary.

## Before Any Overnight Launch

Do not rely on memory. Read:

- `overnight_run_decision.md`;
- `launch_readiness_checklists.md`;
- current manifest/review artifact;
- latest relevant tournament/leaderboard summary;
- current trainer cadence source of truth.
