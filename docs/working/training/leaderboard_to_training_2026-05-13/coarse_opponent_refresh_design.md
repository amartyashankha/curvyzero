# Coarse Opponent Refresh Design

Date: 2026-05-14

## Plain Goal

Let a training run occasionally change the opponent assignment it will use for
future self-play collection, without making the trainer read live tournament
state on every step.

This is not an every-episode or every-env-step refresh. The intended first
target is coarse:

```text
operator/Coach updates run-control intent
-> Coach materializes a concrete assignment
-> trainer notices at a coarse boundary, currently every 2000 learner train iterations
-> next collection batch uses the new assignment
```

## Current Truth

Implemented today:

- Current restart default cadence is
  `CURVYTRON_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER = 2000` in
  `src/curvyzero/contracts/curvytron.py`. The earlier `50`-iteration cadence
  was a design probe, not the active restart default.

- `stable_slots_v1` can turn a leaderboard snapshot into an immutable
  `assignment.json`;
- the trainer can launch with `opponent_assignment_ref`;
- the env can load frozen checkpoint opponents, blank/no-op opponents, and
  proactive wall-avoidant opponents from a mixture.
- local trainer helpers now build DI-engine `reset_param`, check coarse
  collect-boundary buckets, reset collector envs, verify `ready_obs` plus
  `last_reset_info`, and block collection if the reset cannot be proven.
- `_run_visual_survival_train` can now run a direct pending-assignment refresh
  path with `opponent_assignment_refresh_interval_train_iter` and
  `opponent_assignment_refresh_ref`.
- The direct path writes `opponent_assignment_refresh_events.jsonl` and records
  the last refresh events in `summary.json`.
- If refresh is enabled but the LightZero collector hook cannot be installed,
  the trainer fails before calling `train_muzero`.
- A tiny Modal proof, `refresh-e2e-smoke-20260514/train-refresh-d`, completed
  with assignment A at launch and assignment B as the pending refresh. It wrote
  one `applied` event, proved both envs ready with B, and telemetry rows changed
  from A to B after LightZero's initial random-collect startup rows.

Missing today:

- no run-control Modal Dict API for "run X wants slot recipe generation N";
- no Coach materializer that reads that run-control record and writes the next
  assignment for that run;
- no run-control/`ready.json` verifier feeding the running trainer yet;
- no committed `applied.json` handoff back to run-control; the current direct
  path has a local JSONL event log inside the attempt train folder;
- no profile showing the cost of loading a replacement frozen checkpoint
  opponent.

Newest correction:

- The live mutable object is the Modal Dict run-control record.
- The trainer should not materialize slots from a leaderboard.
- The trainer should not poll per env step, per episode, or inside the
  collector loop.
- The trainer may poll a small pending-assignment pointer at a bounded
  collection boundary, initially every about 50 learner iterations.

## Correct Boundary

The mutable object is the run-control record, not the assignment used by a
training transition.

```text
Modal Dict / run-control = proposal pointer and mutable operator intent
Volume ready marker = committed materialization proposal
Volume applied event = trainer actually changed state
env worker ack = every worker reports the new state
```

The trainer should not read Elo, tournament queues, or raw leaderboard state.
It may read a small Coach-owned run-control pointer or pending assignment
pointer at a coarse training boundary.

Assume Dict and Volume have no shared ordering. The trainer must not believe
"Dict says it exists" until it reloads Volume, reads `ready.json`, and verifies
the ready marker plus assignment hash.

## Coarse Boundary Hypothesis

The likely boundary is around the stock LightZero collect/train rhythm, not
inside one env step. The first concrete target is:

- check at most once every 50 learner train iterations;
- only apply before a future self-play/collector batch is built;
- never change the opponent for transitions already collected;
- record which assignment id was active for collection metadata/eval/GIF.

The exact hook point is still under investigation. Candidate places:

1. a collector wrapper that refreshes assignment just before `Collector.collect`
   when `train_iter` has crossed the interval;
2. a learner hook that marks refresh as due after enough `BaseLearner.train`
   calls, then the next collector call applies it;
3. a checkpoint/save hook only if its cadence is deliberately aligned with
   refresh timing;
4. a new-attempt/resume boundary if live in-process refresh is too brittle.

Current best hook:

```text
LightZero loop:
  evaluate maybe
  Collector.collect(train_iter=learner.train_iter)
  replay_buffer.push_game_segments(...)
  learner.train(...) many times
  repeat
```

So the first stock-LightZero target is a wrapper around `Collector.collect`.
At the start of the wrapper, before the original collect call, the trainer may:

1. check whether enough learner iterations have passed;
2. read one small pending-assignment pointer;
3. resolve and verify the pending immutable assignment ref;
4. force-reset the collector envs with the new opponent mixture;
5. call the original `collect`.

This keeps the refresh outside one env step and outside one replay batch.

Important: a wrapper before `Collector.collect` is not enough if it only mutates
state. LightZero's collector reads `ready_obs` at the start of `collect`, and
that `ready_obs` was created by an earlier env reset. The safe wrapper must
apply the new mixture to every env worker, then force-reset, then wait for fresh
ready observations before it calls the original `collect`.

## Simple V0 Rule

Prefer a boring pause-and-reset refresh:

```text
start of Collector.collect
-> if refresh is not due, do nothing
-> if refresh is due, verify the prepared immutable assignment
-> reset every collector env with the new mixture and assignment context
-> prove every env is ready under that assignment
-> collect
```

The check interval can start at 50 or 100 learner iterations. The exact number
is a tuning knob; the safety rule is more important than the number.

Fallback rule:

- If the pending assignment is missing, stale, hash-mismatched, or not visible
  on Volume yet, keep the old assignment and write a visible `kept_previous` or
  `failed` refresh event.
- If no env workers have been changed yet, continuing with the old assignment is
  safe.
- If some env workers may have changed but the trainer cannot prove every worker
  changed and reset, do not collect that batch. That is the split-brain case.
  Rebuild/restart the envs or end the attempt cleanly.

Current direct-path behavior:

- pending assignment load failure keeps the old assignment and retries on the
  next collect in the same bucket;
- a pending assignment without `assignment_sha256` keeps the old assignment and
  retries on the next collect in the same bucket;
- an unchanged assignment advances the bucket and collects normally;
- a changed assignment resets all collector envs, proves every env reports the
  new assignment id/ref/hash/refresh index, resets collector policy/stat state,
  writes an `applied` event, then collects;
- if proof fails after reset, the trainer raises before collecting.

Do not add asynchronous pre-loading in the first version unless a profile proves
checkpoint loading cost is the bottleneck. It is allowed later, but it should
not make the first safety path harder to reason about.

## Policy Loading Question

Refreshing an assignment may mean:

- no model load: blank/no-op or proactive wall-avoidant entry;
- same checkpoint opponent as before;
- new frozen LightZero checkpoint opponent, which may require loading weights.

The implementation should avoid loading weights in a hot env step. The preferred
shape is to start/perform the load at the coarse boundary, then make future
collection use the new policy. If this load is expensive, profile it and decide
whether to pre-load in a helper object before swapping the active assignment.

Important env detail:

- The active frozen opponent object is `self.opponent_policy` in
  `CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv`.
- The current env also keeps loaded opponent objects by mixture entry name.
- Reusing a slot name such as `recent` with a different checkpoint can keep the
  old loaded opponent unless the refresh clears or re-keys those objects.
- A safe refresh must change the mixture, clear selected-entry state, clear
  loaded opponent objects, and reset before the next collection.

Subprocess env detail:

- DI-engine subprocess env managers run envs in child processes.
- The safest V0 control channel is `env_manager.reset(reset_param)` at the
  collection boundary, with reset parameters carrying the resolved opponent
  mixture.
- Local source inspection shows `reset_param` is passed as keyword args to each
  child env reset. That means no custom `method_name_list` setter is needed for
  the first version.
- If a separate env setter is needed, it must be a public env-manager method for
  subprocess workers; mutating main-process env refs does not update children.
- If that cannot be proven for the stock subprocess manager, use a controlled
  resume/new-attempt refresh instead of live in-process refresh.

## First Design Sketch

1. Coach writes or updates `run_control:<run_id>` in Modal Dict.
2. Coach materializes that intent into immutable assignment artifacts on Volume:
   `assignment.json`, `audit.json`, and `refresh.json`.
3. Trainer is launched with a run-control/pending-assignment pointer plus an
   initial `opponent_assignment_ref`.
4. Every coarse interval, trainer checks whether the pending assignment changed.
5. If changed and valid, trainer prepares the new opponent assignment before the
   next collection batch.
6. Trainer records an applied-refresh row with old assignment id, new assignment
   id, hash, iteration, and timing.

There is also a simpler direct path for near-term overnight runs:

```text
launch with initial opponent_assignment_ref=A
launch with opponent_assignment_refresh_ref=B
launch with opponent_assignment_refresh_interval_train_iter=50  # historical test cadence, not current restart18 default
-> trainer checks B at the collect boundary
-> if B differs from A and resolves cleanly, reset/prove/apply before collect
```

This direct path is not the final run-control interface, but it is now covered
by local trainer-level tests and one tiny Modal train proof.

## V0 Run-Control Pointer

The trainer-facing Dict value should be small. It should not contain a
leaderboard or slot-selection program.

```json
{
  "schema_id": "curvyzero_training_run_control/v0",
  "run_id": "train-run-id",
  "attempt_id": "attempt-id",
  "generation": 12,
  "updated_at": "2026-05-14T00:00:00Z",
  "opponent_refresh": {
    "enabled": true,
    "min_check_train_iters": 50,
    "pending_assignment": {
      "refresh_index": 3,
      "target_run_id": "train-run-id",
      "target_attempt_id": "attempt-id",
      "materialization_id": "mat-003",
      "ready_ref": "training/.../ready.json",
      "ready_sha256": "sha256...",
      "assignment_ref": "training/.../assignment.json",
      "assignment_sha256": "sha256...",
      "recipe_generation": 8,
      "recipe_hash": "sha256...",
      "source_snapshot_ref": "tournaments/.../public/latest.json",
      "source_snapshot_sha256": "sha256...",
      "created_at": "2026-05-14T00:00:00Z"
    },
    "applied_assignment": {
      "refresh_index": 2,
      "assignment_ref": "training/.../assignment-old.json",
      "assignment_sha256": "sha256..."
    }
  }
}
```

Trainer applies only if:

- `schema_id`, `run_id`, and `attempt_id` match;
- `generation` is newer than the last applied/read generation;
- `pending_assignment.refresh_index` is newer than the last applied refresh;
- `ready.json` exists, hash-checks, and points at the same assignment/hash;
- the assignment file exists on Volume;
- the assignment hash matches;
- all frozen checkpoint refs are immutable `iteration_N.pth.tar` files;
- the assignment resolves through the existing opponent-mixture resolver.

If any check fails, keep the old assignment and write a failure event.

`ready.json` is written last by Coach/materializer:

```json
{
  "schema_id": "curvyzero_opponent_assignment_ready/v0",
  "run_id": "train-run-id",
  "attempt_id": "attempt-id",
  "refresh_index": 3,
  "materialization_id": "mat-003",
  "assignment_ref": "training/.../assignment.json",
  "assignment_sha256": "sha256...",
  "audit_ref": "training/.../audit.json",
  "audit_sha256": "sha256...",
  "recipe_generation": 8,
  "recipe_hash": "sha256...",
  "source_snapshot_ref": "tournaments/.../public/latest.json",
  "source_snapshot_sha256": "sha256..."
}
```

Applied events are append-only Volume facts. Resume should ignore Dict ack and
recover the active assignment from the highest valid applied event for the same
run and attempt.

## Red-Team Checklist

Highest risks:

- Partial write: Dict points at an assignment before the Volume write/commit is
  visible.
- Stale pointer: trainer sees an old pending assignment after a newer one exists.
- Attempt race: two attempts with the same run id acknowledge the same pending
  assignment.
- Split brain: some subprocess envs update and others keep the old mixture.
- Mixed batch: refresh occurs after `Collector.collect` has started.
- Old loaded opponent: refreshed slot name reuses a previously loaded frozen
  checkpoint object.
- Hash mismatch: assignment JSON changes after the Dict pointer was written.
- Replay ambiguity: replay buffer contains data from several assignments without
  enough metadata to explain learning curves.
- Lazy load stall: frozen checkpoint load happens on the first env action rather
  than at the coarse boundary.
- Boundary leak: trainer starts reading leaderboard/tournament state instead of
  the pending assignment pointer.
- Old `ready_obs`: LightZero collector reads `self._env.ready_obs` at the start
  of collection, so a refresh must reset every env and prove the new ready
  observations are under the new assignment before collection starts.
- Eval lag: LightZero evaluates before collect in the stock loop. If refresh is
  only applied in the collect wrapper, eval at that train iteration may still
  use the previous assignment unless eval envs get a separate refresh path.

Required invariants:

- Dict write is last; Volume assignment and audit files are written first.
- Trainer ack includes run id, attempt id, generation, refresh index, assignment
  ref, assignment hash, train iteration, and decision.
- A failed refresh records why it kept the old assignment.
- No control-plane read happens inside env `step`, env `reset` selection, or the
  collector's inner step loop.
- Each collection batch has one active assignment id.
- If env-manager reset cannot update all envs, the refresh is rejected.
- If envs update but the applied-event write fails, do not collect. Rebuild,
  resume, or fail closed.

Safe apply order:

```text
1. Trainer polls Dict at coarse boundary.
2. Reload Volume.
3. Validate ready marker hash.
4. Validate assignment hash and schema.
5. Update all collector env workers before collect.
6. Verify every worker reports the new assignment id/hash.
7. Write committed applied event to Volume.
8. Only then allow the next collection batch.
9. Optionally update Dict applied ack.
```

Local tests now cover:

- helper bucket behavior around 49/50/51/100 learner iterations;
- exact LightZero `collect(n_episode=None, train_iter=0, ...)` argument parsing;
- deep-copied `reset_param` for every env id;
- all-env proof using `ready_obs` and `last_reset_info`;
- no-op when refresh is not due;
- unchanged assignment hash does not reset envs;
- bad pending assignment before reset keeps old assignment and retries later;
- reset-proof failure blocks the original collect call;
- env reset immediately reports the refreshed assignment in `last_reset_info`;
- reused frozen slot names clear old loaded opponent objects.

## Non-Goals

- No per-step Modal Dict reads.
- No raw tournament/Elo reads inside the trainer.
- No silent in-place overwrite of `assignment.json`.
- No mid-batch or mid-transition opponent mutation.
- No reward recipe refresh in this first path.

## Open Questions

- Can stock subprocess env reset safely receive a new opponent mixture for all
  child envs before a collect call?
- How expensive is frozen checkpoint opponent loading?
- Does the background eval/GIF poller need to follow the latest assignment, or
  should it evaluate each checkpoint with the assignment active when that
  checkpoint was saved?
- Should the first V0 use live in-process refresh, or prove the same contract
  through new attempts/resume first?

## Immediate Research Lanes

- LightZero loop boundary: find the hook before collection.
- Opponent policy object: identify exactly what must be swapped for blank,
  proactive, and checkpoint opponents.
- Run-control contract: decide the smallest Modal Dict value and Volume refresh
  record.
- Tests/profile: define a fake-LightZero regression and one tiny end-to-end
  smoke.
