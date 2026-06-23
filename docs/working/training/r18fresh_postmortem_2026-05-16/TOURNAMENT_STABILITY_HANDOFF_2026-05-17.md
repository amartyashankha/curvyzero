# Tournament Stability Handoff - 2026-05-17

This is the deeper handoff for why the automated tournament loop has needed too
much manual repair. It should be read with
`WHAT_THE_HELL_IS_GOING_ON_HANDOFF_2026-05-17.md`.

## Plain English Summary

14:52 EDT update: the immediate reducer block moved. `round-000036` published,
and trainer export generation `20` now points at `round-000036`. The next
backlog is `765` newer checkpoints in intake. A `round-000037` attempt for the
`4192` checkpoint pool was skipped with zero output, and a fresh bounded drain
was spawned as `fc-01KRVMGPC5EZ8THYHQXPDV0CP4`.

The lesson did not change: durable state and live Modal workers need explicit
ownership and lifecycle records. A skipped or orphaned batch must not silently
block newer checkpoint pools, and a deployed-code fix must not be confused with
a repaired live manifest.

The loop has two kinds of state:

- durable state: named Modal Volumes, Dicts, Queues, rating inputs, rating
  outputs, intake manifests, checkpoints;
- live compute: currently running Modal function calls and workers.

Redeploying code changes future containers. It does not erase durable state, and
it does not necessarily kill already-accepted old workers. Stopping the app kills
live workers, but durable v2 Volumes and Dicts remain.

The recent failure was exactly this split:

- `round-000035` was marked skipped in durable state because it had the wrong
  checkpoint pool;
- but Modal logs still showed old `round-000035` game workers running and
  timing out;
- those workers could consume capacity while state looked ready to move on.

The immediate repair was to stop the deployed tournament app, redeploy current
code, and spawn one fresh drain. That produced `round-000036`, which correctly
covers all `3427` intake checkpoints and is bounded to `300` pairs / `6300`
games.

13:12 EDT update: `round-000036` is correct and alive, but it has not published
yet. The normal path waits for all `6300` games. Logs show some games can run
for more than an hour, so a few stragglers can block publication. A direct
detached partial reduce was spawned as `fc-01KRVEMQYPK41S703PJ7PB5DKA`, but it
stayed pending in a detached local-entrypoint app. The deployed tournament app
and detached local-entrypoint app were stopped, the patched app was redeployed,
and a deployed-app partial reduce was spawned as
`fc-01KRVF11V4F7NVABHYDAWYY8S9`.

Follow-up: another concrete control-plane starvation bug was found. Game worker
warm-container settings were very large (`100/400` for single-game workers and
`500/500` for shard workers). That can reserve hundreds of game workers after a
deploy, before reduce/control work starts. Current code lowers both game paths
to `min=0`, `buffer=16`. This keeps parallel game fanout available while
avoiding idle warm-worker starvation.

Follow-up: `curvytron_rating_reduce` was also oversized for the immediate
partial-reduce job (`2 CPU / 8GB`). Status functions could run while reduce
calls stayed pending without a task id. Current code lowers reduce to
`1 CPU / 2GB`; if a future full reduce needs more memory, raise it deliberately
after measuring, not by default.

Same cleanup now applies to rating progress/provisional functions: they also
move from `2 CPU / 8GB` to `1 CPU / 2GB` so diagnostic/recovery paths have a
better chance of starting under load.

Reduce `max_containers` is also raised from `3` to `20`. Reason: after several
cancelled pending reduce calls, a three-slot lane can become its own bottleneck.
This is only for reduce/publish work, not game fanout.

13:55 EDT update: this was still not enough. Reducer and rescue reducer calls
stayed `PENDING` with no task id. Status/control calls can run, so the problem
is more specific than "Modal is down." Additional live changes:

- reducer resources lowered again to `0.25 CPU / 1GB`;
- tournament game worker warm buffers lowered to zero;
- duplicate reducer body collapsed into one helper;
- added `curvytron_feedback_loop_reduce_rescue`, a reducer entrypoint on the
  same volume set as the status/control lane;
- current active rescue call:
  `fc-01KRVH57NGQY1VZDSVC8N65886`.

Do not treat this as fixed until a reducer call has a task id, returns, and
latest rating advances to `round-000036`.

## Current Verified State

- Latest published rating: `round-000034`.
- Latest published rating checkpoint count: `2739`.
- Intake checkpoint count: `3427`.
- Newer checkpoints waiting for current active rating: `688`.
- Bad batch: `round-000035`, skipped; old live workers were killed by app stop.
- Current active batch: `round-000036`.
- Current active batch checkpoint count: `3427`.
- Current active batch pair count: `300`.
- Current active batch game count: `6300`.
- Current active batch has real game output.
- Trainer export generation `17` from `round-000034` has been consumed by
  trainers: `39218` provider-ok rows, `0` provider-false rows, `48 / 136`
  latest-applied target assignments.

## Root Causes

1. **Live batch input is immutable.**
   Once a rating batch writes its `input.json`, it does not grow when more
   checkpoints arrive.

2. **The drain can spawn from a stale manifest.**
   Checkpoint discovery/tick updates the manifest. Drain reads the manifest. If
   drain runs before the manifest is refreshed, it can build the wrong desired
   pool.

3. **A skipped durable batch does not automatically cancel old workers.**
   Skipping `round-000035` fixed state, but old game workers kept running until
   the app was stopped.

4. **Status was too count-oriented.**
   Counts like queue length, latest rating count, and probe output were useful,
   but they did not fully prove that every active worker matched the desired
   pool.

5. **The control plane treated active work as blocking before proving it was
   current.**
   A live active batch should block only if it is current, bounded, and useful.
   Obsolete or already-rated batches should be droppable.

## Permanent Fix Options

### Fix 1: Refresh Intake Inside Drain

Before `curvytron_checkpoint_intake_drain` builds the rating spec, it should
refresh or rediscover the live checkpoint pool using the same logic as the
subscriber/tick path.

Why it helps: prevents old-pool batches like `round-000035`.

Tradeoff: every drain does more work. This is worth it for live correctness.

### Fix 2: Add A Spawn Invariant

Before spawning a rating loop, check:

- desired checkpoint count;
- intake checkpoint count;
- latest rating checkpoint count;
- desired pool hash;
- active pool limit;
- pair cap;
- games per pair.

If live intake has newer checkpoints and the desired spec does not include them,
do not spawn. Return a clear reason such as `manifest_refresh_required`.

Why it helps: catches stale manifests before work launches.

Tradeoff: can delay a batch, but avoids wasting thousands of games.

### Fix 3: Make Active Batch Blocking Conditional

An active batch should block the next drain only when:

- its input checkpoint pool includes the newer intake checkpoints it is supposed
  to rate;
- it is bounded by the live config;
- it has fresh output or is not old enough to call stale.

Otherwise recovery should mark it skipped/dropped and continue.

Why it helps: bad active state stops blocking the whole feedback loop.

Tradeoff: needs careful liveness checks so we do not skip useful live work.

### Fix 4: Track Worker Function Calls Per Batch

Record the rating-loop call id, rating-round call id, and game worker call ids
or a compact call-graph snapshot for each active batch.

Why it helps: status can tell whether old workers are still alive after a batch
is skipped.

Tradeoff: more metadata and some Modal API calls.

### Fix 5: Add A Cleanup Runbook / Tool

Add an operator command that says:

- which app id is deployed;
- which active batch it is running;
- whether old skipped batches still have live workers;
- whether stopping/redeploying is the right repair;
- exactly what state will and will not be lost.

Why it helps: avoids guessing from logs and avoids repeated manual Volume
digging.

Tradeoff: this is operational hardening, not a replacement for code guards.

### Fix 6: Per-Checkpoint Lineage Ledger

For each checkpoint, record stages:

`discovered -> accepted -> enqueued -> drained -> scheduled -> rated -> exported -> trainer_loaded`.

Why it helps: lets us answer “where did this checkpoint get stuck?” directly.

Tradeoff: more artifacts and more write paths, but it turns vague debugging into
state inspection.

### Fix 7: Active Batch Ownership Record

For each active game batch, write a small ownership record:

- tournament id;
- rating run id;
- batch id;
- checkpoint count and pool hash;
- pair count and game count;
- drain call id;
- rating-loop call id;
- rating-round call id;
- reduce/recovery call id if any;
- created time and last observed output time.

Why it helps: we can answer "who owns this batch?" and cancel or inspect the
right Modal call without guessing from logs.

Tradeoff: it adds a little state bookkeeping, but it is the cleanest way to
separate durable state from live compute.

### Fix 8: Partial Reduce / Straggler Policy

The current game cap is intentionally very high. That is fine for gameplay, but
the rating loop must not wait forever for every single game. A batch should be
allowed to publish from completed games after a configured delay, then later
either merge stragglers into a fuller rating or leave them as late evidence.

Why it helps: one unusually long game cannot freeze the feedback loop.

Tradeoff: a partial rating is less complete than the full 21-games-per-pair
target. It is still better than no rating when the alternative is a frozen
feedback loop.

### Fix 9: Control-Plane Capacity Reservation

Game workers should not reserve so many warm containers that reduce/status
functions cannot start. Keep game warm buffers small, and if needed split
control/reduce into a separate app or deployment lane.

Why it helps: the tournament can still run many games, but publishing and
operator checks keep a path to run.

Tradeoff: the first wave of game workers may cold-start more often, but that is
better than freezing the feedback loop.

### Fix 10: Separate Publish App Or Hard Control Reservation

If the main tournament app can still queue reducer calls while status calls run,
the publish path should move to a separate tiny app or a hard-reserved control
lane that never shares warm workers or game fanout configuration.

Why it helps: publishing ratings is the feedback-loop heartbeat. It should not
compete with gameplay or scheduled subscriber ticks.

Tradeoff: more deployment plumbing, but it makes the most important operation
independent and easier to reason about.

## Next Checks Before Saying Stable

1. `round-000036` keeps producing game output. One liveness sample is not enough
   to call it healthy.
2. `round-000036` completes/reduces/publishes.
3. Latest rating advances from `round-000034` to `round-000036`, with
   checkpoint count `3427`.
4. `refresh-if-ready` writes a trainer export from `round-000036`.
5. `trainer-proof` shows real trainers loading the `round-000036` export, with
   `provider_false=0`.
6. A later intake check shows new checkpoints after `3427` are picked up by the
   next bounded batch automatically.
7. The next batch after `round-000036` does not repeat the old-pool bug.

## Minimal Implementation Sequence

1. **Status/control consistency.**
   Every status that says recovery is needed must map to a control decision that
   can actually run recovery. Cover stale output, zero-output stale,
   obsolete/smaller pool, and oversized legacy batches.

2. **Drain-side manifest refresh.**
   Before drain spawns a rating loop, refresh or rediscover the live checkpoint
   pool. If the refreshed manifest has more refs than the desired rating spec,
   do not spawn; return `manifest_refresh_required`.

3. **Active batch contract validation.**
   Store and compare: pool hash, checkpoint count, pair count, game count,
   pair-selection mode, pair cap, active-pool limit, and games per pair. Only a
   current, bounded, useful batch should block the next drain.

4. **Worker/orphan visibility.**
   Persist rating-loop and rating-round call ids for each active batch. Status
   should be able to say whether skipped/dropped batches still have live workers.

5. **Incremental progress index.**
   Full recovery/progress scans can time out. Maintain a compact progress
   artifact while games write summaries so status can answer "how many of 6300
   games are done?" without scanning the entire tree.

6. **Partial reduce policy.**
   An active batch with real output should be able to publish partial ratings
   after an age threshold, so long games do not block the loop forever.

7. **Skipped means cancel, not just hide.**
   A skipped/dropped batch should write a cancel marker, and game workers should
   check that marker before running and before writing summaries. Otherwise
   state can say "skipped" while old workers continue burning compute.

8. **Recovery terminal states.**
   Use explicit terminal states like `dropped_obsolete_pool`,
   `dropped_oversized_legacy_batch`, and `dropped_stale_zero_output` instead of
   a vague running/skipped state.

9. **Persist round call ids and add cautious cancellation tooling.**
   Store the rating-round call id. An operator command can then inspect the
   call graph and, with explicit confirmation, cancel the exact stale call. Do
   this only for known tournament calls, not broad app-wide cleanup.

10. **Per-checkpoint lineage.**
    Track each checkpoint through discovery, intake, scheduling, rating, export,
    and trainer load. This is the direct answer to “where did this checkpoint get
    stuck?”

11. **Control-plane capacity reservation.**
    Keep game worker warm buffers small, and consider a separate reduce/control
    app if live fanout still starves publishing.

## What Not To Do

- Do not say “queue length is zero” means the loop is caught up.
- Do not say “a batch is active” means the right checkpoints are being rated.
- Do not say “trainer export generation increased” unless the export source
  rating matches the latest rating.
- Do not spawn duplicate drains while a current, bounded, useful batch is
  active.
- Do not rely on redeploy alone to repair persistent state.
