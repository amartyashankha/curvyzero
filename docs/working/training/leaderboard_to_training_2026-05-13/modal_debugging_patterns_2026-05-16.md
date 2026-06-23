# Modal Debugging Patterns For Leaderboard-To-Training

Date: 2026-05-16

Scope: web-research/debugging lane only. No runtime state touched.

Target loop:

```text
trainer writes checkpoints
-> subscriber/intake discovers them
-> tournament rates them
-> tournament publishes leaderboard snapshot + Dict pointer
-> controller/materializer writes immutable assignment + refresh pointer
-> long-running trainer refreshes opponents at safe boundaries
```

## Official Modal Patterns

Use deployed apps for autonomous scheduled services, and treat detached
ephemeral apps as a launch-mode choice that must be explicit. Modal documents
that `modal run`/`app.run` create ephemeral apps that stop when the caller exits
unless `--detach` is used, while deployed apps persist until stopped and
scheduled functions run from deployed apps.

Sources:

- <https://modal.com/docs/guide/apps>
- <https://modal.com/docs/reference/cli/run>

For background work, prefer `.spawn()` when the caller should not block, persist
the returned FunctionCall id, and poll or inspect later. Modal's job-queue docs
describe the deploy -> `Function.spawn()` -> `FunctionCall.get()` pattern, and
the `FunctionCall` reference exposes `get(timeout=0)` polling, `from_id()`,
`get_call_graph()`, and `cancel()`.

Sources:

- <https://modal.com/docs/guide/job-queue>
- <https://modal.com/docs/reference/modal.FunctionCall>

For long-running training, make work checkpointed, retryable, and resumable.
Modal's long-training example reduces the pattern to: save checkpoints to a
Volume, resume from the latest checkpoint on startup, and add retries. Modal
also documents a default 5-minute Function timeout, configurable up to 24 hours,
and warns that long-running training Functions should tolerate preemption by
saving work frequently and being safely retryable.

Sources:

- <https://modal.com/docs/examples/long-training>
- <https://modal.com/docs/guide/timeouts>
- <https://modal.com/docs/guide/preemption>
- <https://modal.com/docs/guide/retries>

Use Volumes as durable file truth, but respect their consistency model. Writes
become visible to other containers after commit, and existing containers must
reload to see newer committed state. Background commits happen every few
seconds and a final commit happens on container shutdown, but explicit commit
is still the right debugging boundary for cross-container handoff. Concurrent
writes to the same file are last-writer-wins, reload can fail when files are
open, and the reloading container sees the Volume as empty while reload is in
progress.

Sources:

- <https://modal.com/docs/guide/volumes>
- <https://modal.com/docs/reference/modal.Volume>

Use Queues for active wakeups, not durable history. Modal Queues are FIFO,
partitioned, blocking by default, and good for asynchronous coordination, but
they are cleared 24 hours after the last `put`, partition/item limits apply,
and persistence is "likely, but not guaranteed."

Sources:

- <https://modal.com/docs/guide/queues>
- <https://modal.com/docs/reference/modal.Queue>

Use Dicts for compact distributed state such as pointers, claims, active keys,
and operator intent, not bulky artifacts or historical truth. Modal Dict values
are persisted across redeploys, but mutable values must be explicitly put back,
entries expire after inactivity, and object-size/update limits make them a poor
home for leaderboard or assignment bodies.

Sources:

- <https://modal.com/docs/guide/dicts>
- <https://modal.com/docs/reference/modal.Dict>

Use Modal's app-level logs, dashboard, deployment history, and version
transition semantics when debugging "it worked before redeploy" issues.
`modal app logs` supports time windows, search text, function/function-call id,
container id, stdout/stderr/system filters, and timestamp/container metadata.
Deployment docs say old containers keep handling accepted requests while new
containers start, so overlapping app versions are expected during redeploys.

Sources:

- <https://modal.com/docs/reference/cli/app>
- <https://modal.com/docs/guide/developing-debugging>
- <https://modal.com/docs/guide/managing-deployments>

## Repo Mapping

Current local docs describe the durable/control split this way:

```text
checkpoint Volume -> intake manifest/Queue -> rating loop -> latest.json
-> public leaderboard snapshot + Dict pointer -> immutable assignment
-> control pointer -> same-running-trainer refresh -> provider-ok env telemetry
```

Relevant local docs:

- `docs/working/training/leaderboard_to_training_2026-05-13/dataflow.md`
- `docs/working/training/leaderboard_to_training_2026-05-13/closed_loop_spec.md`
- `docs/working/training/leaderboard_to_training_2026-05-13/FULL_LOOP_PROOF.md`
- `docs/working/training/leaderboard_to_training_2026-05-13/modal_feedback_loop_debugging_followup_2026-05-16.md`

Relevant implementation surfaces:

- Runtime objects: `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament_runtime.py`
- Tournament/intake/controller: `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
- Trainer/checkpoint/refresh hook: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- Pure leaderboard/assignment contracts: `src/curvyzero/training/opponent_leaderboard.py`
- Run artifact helpers: `src/curvyzero/infra/modal/run_management.py`

The current app/storage split matches the Modal guidance well:

| Loop stage | Modal primitive | Durable truth | Debug invariant |
| --- | --- | --- | --- |
| Trainer checkpoint save | Runs Volume | exact `iteration_N.pth.tar` plus sidecar/progress files | checkpoint ref exists after commit/reload and is immutable |
| Subscriber/intake | Dict + Queue + Tournament Volume | intake manifest artifact and tick/progress artifacts | Queue can be empty; manifest must still reconstruct desired refs |
| Rating/tournament | spawned Functions + Tournament Volume | round input/results/ratings/latest/pair history | scheduling is not completion; latest/ratings must advance |
| Public leaderboard | Tournament Volume + Dict pointer | immutable snapshot and `latest.json` on tournament Volume | Dict pointer is repairable cache, not source of truth |
| Assignment materializer | Tournament, control, checkpoint Volumes | assignment/audit files plus refresh pointer | assignment sha matches pointer before trainer can consume |
| Trainer refresh | control/runs Volume reload + env reset | trainer JSONL refresh events and env telemetry | `decision=applied` and provider-ok rows prove real use |

## Robust Patterns To Preserve

The strongest pattern in the repo is the "immutable body, mutable pointer"
shape. Leaderboard snapshots, assignments, audits, rating round files, and
checkpoint files are written as durable artifacts; only compact Dict entries or
small pointer JSON files move. That lines up with Modal's Volume and Dict
contracts: Volume for durable payloads, Dict for compact active state.

The queue is correctly treated as a wakeup channel. Intake stores seen/queued
checkpoint refs in the manifest and has repair logic for missing Queue events,
so a zero queue length is not interpreted as "all work is complete."

The controller writes and commits in a careful order:

1. reload tournament/control/checkpoint Volumes;
2. validate the source rating snapshot;
3. build and commit training-candidate leaderboard snapshot/latest;
4. write and commit assignment/audit bodies;
5. re-check old refresh pointers;
6. rewrite and commit refresh pointers;
7. publish compact leaderboard Dict pointer.

That is the right ordering because trainers only observe the new assignment
after the assignment body has been committed and the pointer sha has been
validated.

The trainer refresh hook is a good safe-boundary implementation. It reloads
Volume state before reading a pending assignment, resolves pointer chains with
sha checks, rejects missing assignment sha, skips unchanged shas, resets
collector envs only at collector boundaries, records `decision=applied`, and
checks the env ready report before training continues.

The run helpers use temp-file-plus-replace JSON writes through
`runs.write_json()`, which is the right local filesystem pattern for avoiding
half-written hot JSON files before Volume commit exposes the result.

## Failure Modes And Stalls

| Symptom | Likely cause | Proof to collect |
| --- | --- | --- |
| Child games or ratings vanish after launch | non-detached ephemeral parent returned; app stopped before child `.spawn()` work finished | app logs/history; `latest.json` and completed game summaries, not only initial scheduling output |
| Queue is empty but new checkpoints are not rated | Queue event expired/lost or drained before rating claim completed | compare durable intake manifest refs, queued refs, latest rating roster refs, and queue partition length |
| Intake repeatedly spawns the wrong pool | stale or too-broad claim key; continuation not tied to pool hash/mode | inspect claim Dict payload, pool hash, `continue_from_latest`, and checkpoint count |
| Tournament appears active but leaderboard is stale | running round has not reduced to `latest.json`, or web/container cache has stale Volume view | direct Volume fetch of round input/results/ratings/latest; look for reload errors in logs |
| Published leaderboard points at bad source | rating snapshot is provisional, missing one-frame metadata, or has unexpected source hash | validate expected round id/index/context/roster/source sha before publish |
| Trainer sees pointer but does not switch | refresh interval not reached, pointer rewrite not committed, reload failed, or pointer sha mismatch | inspect control pointer, assignment body sha, refresh JSONL events, and env ready report |
| `opponent_provider_load_ok=false` | assignment points at missing checkpoint, wrong Volume prefix, mutable ref, or stale control-volume copy | resolve every frozen checkpoint ref from the trainer container's mount view |
| Deployed behavior changes during a run | redeploy overlap; old containers continue accepted work while new version starts | `modal app history`, logs with container/function-call ids, and artifact `app_name`/writer metadata |
| UI/subscriber reload produces confusing missing files | Volume reload in progress or `volume busy` due open files | log reload error, keep stale read-only cache, retry after closing handles |
| Long trainer restarts or stalls near timeout/preemption | Function timeout, preemption, crash-loop backoff, or non-reentrant resume state | checkpoint cadence, latest attempt/progress files, app logs with function-call id, and resume audit |

## Debugging Ladder

Use this order when diagnosing a stalled feedback loop:

1. Prove app lifetime.
   Check whether the work is on a deployed app or a detached ephemeral app.
   For background tournament work, a successful submit response is insufficient
   if the parent app can die before child workers finish.

2. Prove exact checkpoint visibility.
   Fetch or list the exact `iteration_N.pth.tar` refs from the runs Volume.
   The subscriber should discover immutable iteration files, not a mutable
   "latest" guess.

3. Prove intake durable state.
   Compare the intake Dict record with the tournament Volume manifest. If they
   disagree, prefer the Volume artifact or rebuild the Dict from it.

4. Prove scheduling versus completion.
   A spawned FunctionCall id proves only submission. Completion requires round
   artifacts: input, progress, results/ratings, and final `latest.json`.

5. Prove leaderboard source identity.
   Before trusting a leaderboard, record the rating round id/index, context
   hash, roster/pool hash, and rating snapshot sha. This catches stale or
   smaller-pool publishes.

6. Prove pointer/body consistency.
   Read the assignment pointer, read the assignment body, recompute canonical
   sha, and confirm the pointer's expected sha matches.

7. Prove live trainer consumption.
   The final proof is not "assignment written." It is trainer refresh event
   `decision=applied`, an env ready report for the new assignment sha, and
   env telemetry showing frozen checkpoint opponent rows with
   `opponent_provider_load_ok=true`.

## Operational Recommendations

Keep Volume artifacts as the recovery ledger. Every Dict key and Queue event
used by this loop should be derivable from Volume manifests, rating artifacts,
or control assignment bodies.

Persist every spawned long-running call id in the artifact graph. That makes
`FunctionCall.from_id()`, `get(timeout=0)`, `get_call_graph()`, and app-log
filters useful after the launch terminal is gone.

Prefer deployed scheduled functions for the subscriber, drain, and controller
ticks. Use `modal run --detach` only for explicit experiment lanes, and write
the launch command/mode into the manifest so future debugging knows whether
app lifetime depended on a local client.

Never interpret a Queue drain as rating success. Rating success is the
appearance and advancement of the rating artifacts on the tournament Volume.

Never interpret a leaderboard Dict pointer as truth. It is a cache that should
be repairable from the immutable snapshot/latest artifacts.

Keep refresh-pointers small and sha-guarded. Trainers should continue to
consume immutable assignment bodies at safe boundaries, not query a live
leaderboard during learner updates.

When debugging redeploy-adjacent anomalies, include app version/deployment
history, function-call id, and container id in the evidence packet. Modal's
zero-downtime transition means old and new code can legitimately overlap.

## Minimum Evidence Packet

For a complete "checkpoint reached training again" proof, collect:

- trainer checkpoint ref, size, sha if available, and sidecar metadata;
- intake manifest ref and manifest `seen_checkpoint_refs`/pool hash;
- queue partition and current length, with note that zero length is allowed;
- rating round input ref and latest/rating snapshot ref;
- rating source identity: round id/index, context hash, roster hash, snapshot sha;
- leaderboard snapshot ref, latest ref, Dict pointer key/value;
- assignment ref, audit ref, assignment sha, pointer ref, pointer sha expectation;
- trainer refresh JSONL rows with `decision=applied`;
- env telemetry rows for the assignment sha with provider-ok frozen checkpoint opponents;
- app logs with timestamps, function-call ids, and container ids around every transition.

