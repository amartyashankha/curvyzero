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

Use the plain-language glossary in `README.md` when writing operator summaries.
Do not assume readers know terms like queue-loss repair, stale-claim repair, or
safe refresh.

## Modal Operating Notes

Local Modal skill notes are available at:

```text
/Users/shankha/Downloads/skills-main/modal/SKILL.md
/private/tmp/modal-auto-research-skills/README.md
```

Use them when Modal behavior is part of the task. See
`modal_skills_pointers.md` for the short local index of the larger packet.

Important rules:

- Prefer `modal --help` and command-specific `--help` before guessing.
- Most Modal CLI commands can emit JSON with `--json`; prefer parseable output
  for tools.
- `modal run` creates an ephemeral App. A non-detached `modal run` is not a
  safe parent for background tournament game/rating workers after the local
  command exits.
- If a command spawns child tournament workers and those workers must continue
  after the command returns, use `modal run --detach` or keep the parent command
  alive until the child work finishes.
- Do not treat "round scheduled" or "rating call id returned" as success.
  Verify `latest.json` advanced and completed game summaries exist.
- Production-like fanout should use one deployed App with many function calls,
  not one App per training row.
- Global Modal app code runs locally and in remote containers; keep it light and
  avoid fragile file/env reads at import time.

## Debug Bundle First

Before diagnosing tournament/intake failures by hand, collect the same four
artifacts every time:

- intake manifest;
- latest intake tick/progress;
- rating config;
- rating latest/progress/results.

Then compare the counts:

```text
manifest seen checkpoints -> rating config checkpoints -> rating latest rows
```

If the manifest sees many checkpoint players but rating config/latest sees only
a few, the bug is in intake/continuation before game scheduling. Do not spend
time debugging pair scheduling or game workers until that count mismatch is
understood.

If progress says a round was written but `latest.json` did not advance, game
directories are empty, or no completed summaries exist, check the launch
lifetime first. Logs containing `RemoteError`, `KeyboardInterrupt`, or
`Runner terminated` after a non-detached `modal run` usually mean the local
entrypoint stopped and Modal killed child work.

The helper script is:

```text
scripts/curvytron_tournament_debug_bundle.py
```

Run it on fetched local artifacts before opening raw JSON by hand.

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

## Repair A Missing Leaderboard Pointer

Use this when the immutable public leaderboard snapshot exists on the tournament
Volume, but the compact Modal Dict pointer is missing or stale.

Command shape:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode leaderboard-pointer-repair \
  --leaderboard-id <leaderboard_id>
```

Expected result:

- `pointer_published=true`;
- `pointer_key=current:<leaderboard_id>`;
- `snapshot_ref` points at an immutable snapshot under
  `tournaments/curvytron/leaderboards/<leaderboard_id>/snapshots/...`;
- compact summary reports `active_count`, `provisional_count`, and
  `retired_count`.

What this does:

- rebuilds the live Dict pointer from durable Volume snapshots;
- does not create ratings;
- does not select training assignments;
- does not change trainer state.

Tiny remote smoke already passed for:

```text
leaderboard_id=curvytron-latest212-smoke-20260513
```

Do not treat this as queue/intake repair proof. Queue-loss, stale-claim, and
rating-continuation repair still need their own remote smokes.

## Before Any Overnight Launch

Do not rely on memory. Read:

- `overnight_run_decision.md`;
- `launch_readiness_checklists.md`;
- current manifest/review artifact;
- latest relevant tournament/leaderboard summary;
- current trainer cadence source of truth.
