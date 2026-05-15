# Run Slot Control Design

Date: 2026-05-14

## Plain Goal

After a training run starts, an operator should be able to change what kind of
opponents that run should use next.

The clean split:

```text
Modal Dict = current desired slot recipe for a training run
Volume JSON = exact assignment the trainer actually used
```

The trainer should not turn the Dict into live per-step behavior. The Dict is
control-plane intent. The durable training input is still an immutable
`assignment.json`.

Follow-up design note: this slot recipe can become one section of a broader
per-run control record. That broader record can also carry reward settings such
as survival, bonus, and final-outcome weights. See
`run_reward_control_design.md`.

## Proposed Shape

One Modal Dict stores recipes keyed by training run id.

```text
dict name: curvyzero-training-slot-recipes
key: run_slot_recipe:<training_run_id>
```

Value:

```json
{
  "schema_id": "curvyzero_run_slot_recipe/v0",
  "run_id": "curvy-mix-example",
  "generation": 3,
  "updated_at": "2026-05-14T00:00:00Z",
  "updated_by": {"kind": "operator", "id": "manual"},
  "enabled": true,
  "leaderboard": {
    "leaderboard_id": "curvytron-main",
    "snapshot_ref": null,
    "snapshot_sha256": null,
    "expected_rating_context_hash": "ctx..."
  },
  "materializer": {
    "strategy_id": "stable_slots_v1",
    "profile": "stable_3",
    "sentinel": "blank_canvas",
    "checkpoint_death_mode": "normal",
    "allow_recent_provisional": false
  },
  "weights": {
    "champion": 20.0,
    "recent_strong": 15.0,
    "diverse_challenger": 10.0,
    "sentinel": 2.0
  },
  "refresh": {
    "mode": "checkpoint_boundary",
    "requested_refresh_index": 2,
    "min_seconds": 1800,
    "fallback_policy": "keep_previous_assignment"
  },
  "cached_latest": {
    "assignment_ref": "training/.../assignment.json",
    "assignment_sha256": "sha256...",
    "refresh_index": 1,
    "materialized_at": "2026-05-14T00:00:00Z"
  },
  "notes": "operator-written human note"
}
```

The recipe says what to select. The assignment says what was selected.

## Flow

```text
operator writes recipe to Dict
materializer reads recipe + public leaderboard snapshot
materializer writes assignment.json + audit.json to the training Volume
trainer picks up assignment only at launch/resume/refresh boundary
```

Every assignment audit should record:

- Dict name and key;
- recipe generation;
- recipe hash;
- leaderboard snapshot ref and hash;
- selected slots and fallback reasons;
- refresh index;
- training run id and attempt id.

Also write a small refresh record next to the assignment:

```json
{
  "schema_id": "curvyzero_opponent_assignment_refresh/v0",
  "training_run_id": "curvy-mix-example",
  "attempt_id": "attempt001",
  "refresh_index": 2,
  "recipe_generation": 3,
  "previous_assignment_ref": "training/.../assignment-old.json",
  "new_assignment_ref": "training/.../assignment.json",
  "new_assignment_sha256": "sha256...",
  "source_snapshot_ref": "tournaments/.../snapshot.json",
  "source_snapshot_sha256": "sha256...",
  "decision": "applied",
  "fallback_reason": null,
  "created_at": "2026-05-14T00:00:00Z"
}
```

## Why This Fits The Architecture

This keeps the pieces simple:

- Coach owns the recipe, materialization, assignment timing, and trainer launch.
- The long-running tournament job owns ratings and public leaderboard snapshots.
- The trainer consumes concrete assignments. It does not understand the live
  tournament or the live Dict.

If the Dict is lost, the run is still explainable from Volume artifacts. If a
recipe is changed, the next materialized assignment records the exact recipe
generation that caused it.

## First Implementation

Keep this out of the learner loop.

1. Add a pure recipe validator near `opponent_leaderboard.py`.
2. Add a Coach/operator function to put a recipe for a run id.
3. Add a Coach/operator function to materialize the current recipe into an
   assignment for a run attempt.
4. Store the assignment and audit under the run attempt, using the existing
   assignment writer path.
5. Update the Dict cached pointer only after Volume files are written.
6. Add tests for recipe validation, generation changes, audit fields, and
   missing/stale Dict behavior.

Do not implement automatic periodic refresh until the manual path is boring.

## Failure Policy

If refresh cannot prove the requested recipe and leaderboard snapshot are valid,
keep the previous assignment.

Fail closed for:

- missing recipe;
- stale recipe generation;
- missing or stale leaderboard pointer;
- snapshot hash mismatch;
- rating context mismatch;
- no active rows;
- duplicate checkpoint refs;
- provisional row selected without an explicit recipe flag;
- partial Volume write.

The operator should see the failure in a refresh record. The trainer should not
silently switch to a weaker surprise assignment.

## Open Questions

- Should the recipe point to `leaderboard_id` only, or to an exact snapshot ref?
  Initial answer: recipe points to `leaderboard_id`; the materialized assignment
  records the exact snapshot ref.
- Should the trainer poll for new assignments? Initial answer: no. A refresh
  operator or boundary hook writes a pending assignment, then the trainer swaps
  only at a safe boundary.
- Should recipe changes be versioned in Volume too? Initial answer: yes, once
  automation exists. Dict is convenient, but Volume should keep the history.

## Anti-Patterns

- trainer reads Modal Dict every episode;
- trainer directly reads leaderboard snapshots;
- tournament job writes assignments;
- recipe changes silently overwrite assignment history;
- recipe schema becomes a broad programming language.
