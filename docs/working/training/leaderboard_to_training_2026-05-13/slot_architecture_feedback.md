# Slot Architecture Feedback

Date: 2026-05-14

## Decision

Opponent slots are not live leaderboard cells and not trainer-side logic. They
are plain assignment entries.

The production direction is:

```text
Tournament job publishes leaderboard snapshot
Coach stable_slots_v1 materializer verifies it
Coach writes immutable assignment.json + audit.json
Trainer consumes assignment at launch/resume/checkpoint boundary
```

`slot_rules_v0` should be purged as the production direction. It was useful as a
pressure test, but it is too close to a small policy language. Do not grow it.

The trainer consumes only the materialized assignment. It never polls live
tournament state, Modal Dict, or Queue during learning.

## Run-Scoped Slot Recipe Direction

Slot definitions should become run-scoped operator state in a Modal Dict.

Plain shape:

```text
run_slot_recipe:<training_run_id> -> desired slot recipe / generation
```

This should be treated as mutable control-plane intent, not as training truth.
The safe path is:

```text
Modal Dict slot recipe -> verified materializer -> immutable assignment.json ->
trainer refresh at launch/resume/checkpoint boundary
```

That lets an operator change a run's desired slots after launch while preserving
an auditable record of exactly which assignment the trainer actually used.

See `run_slot_control_design.md` for the proposed value shape and first
implementation plan.

## Modal Primitives

| Primitive | Role |
| --- | --- |
| Modal Volume | Truth. Stores checkpoints, leaderboard snapshots, assignments, audits, and refresh records. |
| Modal Dict | Cache. May point to the latest public snapshot or latest assignment, but must be repairable from Volume. |
| Modal Queue | Wakeup. May notify that checkpoints or snapshots changed, but is never durable state. |
| Subscriber | Repairable scanner. Finds checkpoints/snapshots from Volume and sends wakeups. |

If Dict or Queue is lost, the system must recover from Volume JSON.

## stable_slots_v1

`stable_slots_v1` is a Coach-owned materializer outside the learning loop.

It should:

1. Read one published leaderboard snapshot.
2. Verify the snapshot ref, hash, leaderboard id, and rating context.
3. Pick 3-5 concrete opponents.
4. Write `assignment.json`.
5. Write `audit.json` explaining every choice.

Default slots:

| Slot | Source |
| --- | --- |
| `champion` | top trusted active checkpoint |
| `recent_strong` | trusted recent checkpoint, using `recency.latest_for_run` |
| `diverse_challenger` | trusted checkpoint from a different run when possible |
| `anchor` | optional stable older checkpoint |
| `sentinel` | optional blank or wall-avoidant immortal entry |

The assignment entries are the slots. Hard-coded training pressure, such as
blank canvas or wall-avoidant immortal, should appear as normal assignment
entries.

## Rules

- dedupe checkpoint slots by checkpoint id and checkpoint ref;
- prefer active rows;
- allow provisional rows only when explicitly requested and recorded;
- fail clearly when a required slot cannot be filled;
- record rank, rating, status, run id, checkpoint ref, fallback reason, source
  snapshot ref, and source snapshot hash in `audit.json`;
- do not silently rewrite or overwrite an existing assignment.

## Designs To Avoid

- trainer polling Modal Dict or Queue;
- tournament publisher writing training assignments directly;
- Queue events as truth;
- mutable slot cells in Dict;
- per-episode "top N from leaderboard" lookup;
- growing `slot_rules_v0` into a production mini language.

## Validation State

Done locally:

- parser-compatible assignment output;
- nested `recency.latest_for_run`;
- checkpoint id/ref dedupe;
- context hash mismatch;
- provisional row gating;
- blank sentinel;
- wall-avoidant immortal sentinel;
- audit source snapshot hash and per-slot evidence;
- local CLI default now uses `stable_slots_v1`;
- local CLI can read an already-published leaderboard snapshot directly instead
  of rebuilding from raw rating JSON.
- tournament rating rows now preserve checkpoint recency metadata needed by
  `recent_strong`: run id, attempt id, iteration, mtime, and `latest_for_run`.

Still needed before trusting automatic refresh:

1. Turn the local CLI/operator behavior into a documented production runbook.
2. Rerun a bounded multi-checkpoint smoke after the recency metadata repair:
   - training emits several checkpoints;
   - the tournament job ranks them under one-frame tournament settings;
   - Coach materializes a 3-5 slot assignment with `stable_slots_v1`;
   - `recent_strong` points at the latest useful checkpoint for the watched run;
   - a trainer consumes that exact assignment.

## Feedback To Other Slot Agents

Keep the system boring. Every training run should be able to answer these
questions from Volume JSON alone:

```text
Which assignment did I use?
Which leaderboard snapshot did it come from?
Which materializer profile filled each slot?
Which exact checkpoint ref did each leaderboard slot resolve to?
When, if ever, did I swap assignments?
```

If a design cannot answer those questions without live Dict or Queue state, it
is too live and too fragile.
