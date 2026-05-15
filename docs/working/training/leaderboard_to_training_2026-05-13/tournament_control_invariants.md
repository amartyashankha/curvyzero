# Tournament Control Invariants

## Plain Problem

The current loop is too brittle because discovery, Queue events, claims, rating
runs, and publishing can disagree.

The failure from the May 14 live smoke is the concrete example:

- training wrote real checkpoints;
- intake eventually saw 159 checkpoint players;
- the rating run that completed only rated 9 checkpoint players;
- a later fresh claim existed while the Queue was empty, so the larger pool did
  not automatically become the public rating.

That is a control-plane problem, not a game-worker problem.

A later May 14 launch-lifetime failure had a different shape:

- round input/progress was written;
- child game workers were spawned;
- the local non-detached `modal run` entrypoint/app stopped;
- logs showed `RemoteError`, `KeyboardInterrupt`, and `Runner terminated`;
- the Volume had empty game directories and no completed summaries.

That is a launch-lifetime problem. The workers did not get a long-lived parent.

## Source Of Truth

Durable Volume JSON is truth.

Modal Dict and Modal Queue are only coordination:

- Queue means "wake up and reconcile";
- Dict means "cache this pointer" or "lease this work briefly";
- neither should be the only place that says which players exist, which work is
  complete, or which pool is allowed to publish.

## Core Invariants

- Player admission is monotonic. Once a checkpoint player is admitted, it is not
  silently dropped.
- A rating result is valid only for the exact player pool it rated.
- A 9-player rating can never satisfy a desired 159-player leaderboard.
- Claims are leases with expiry, not ownership forever.
- Workers derive work from durable manifests, not from drained Queue contents.
- Duplicate Queue events and repeated discovery must converge to the same
  manifest.
- If Modal Dict forgets a manifest, explicit tournament/rating operations should
  rebuild it from the durable Volume manifest.
- Claims must be scoped to the desired player pool. A claim for a 9-player pool
  cannot block a later 159-player pool.
- A command that spawns child tournament workers must either stay alive until
  the children finish or run detached. Non-detached `modal run` is not a safe
  parent for background work that should continue after the command exits.
- "Round scheduled" is not success. Success requires durable rating output:
  `latest.json` advanced and completed game summaries exist.
- Publishing points only at complete immutable rating artifacts.
- Training remains isolated: the trainer writes checkpoints and consumes frozen
  assignments; it does not poll live tournament state while learning.

## Better Shape

Use durable desired state plus immutable rating attempts.

Records:

- `PlayerRegistry`: append-only checkpoint players with stable player ids,
  checkpoint refs, metadata, and status.
- `PoolVersion`: exact eligible player set at a point in time.
- `RatingRun`: immutable attempt for one pool version.
- `GameShard`: deterministic work item for a rating run.
- `LeaderboardPointer`: tiny pointer to a completed rating run.

Legal transitions:

```text
player: undiscovered -> discovered -> eligible -> retired
rating run: planned -> running -> completed -> published
rating run: planned -> running -> abandoned
shard: pending -> leased -> completed
shard: leased -> expired -> pending
```

## Recommended Next Patch Direction

Do not keep growing a global `rating_claim` concept.

Next cleanup should move toward:

1. A durable manifest/pool version that records the exact desired player set.
2. A reconciler that reads that manifest and creates or resumes the correct
   rating attempt.
3. Claims scoped to immutable work items, such as `rating_run_id + shard_id`,
   with expiry and heartbeat.
4. Publisher validation that the published leaderboard player count matches the
   intended pool, unless missing players are explicitly excluded with reasons.

## Tests That Should Exist

- Start from 9 discovered players, run a rating, then discover 159. Assert the
  system creates or resumes a 159-player rating and does not treat the 9-player
  result as current truth.
- Delete Queue messages after discovery. Assert reconciliation still creates
  work from durable state.
- Replay the same checkpoint events many times. Assert the manifest has no
  duplicate players.
- Create a fresh claim from a partial pool, then expand the pool. Assert the new
  pool can still create work.
- Expire a worker lease. Assert another worker can finish the shard.
- Publish after a partial rating. Assert publication fails unless the snapshot
  explicitly marks omitted players and says why.

## Minimal Observability

Every live dashboard/debug bundle should show:

- discovered player count;
- eligible player count;
- desired pool version and player count;
- active rating run id and player count;
- pending/running/completed shard counts;
- latest completed round in `latest.json`;
- count of completed game summaries and empty game directories;
- published pool version and player count;
- lag between latest eligible pool and published pool.

The key alert:

```text
eligible_player_count > published_player_count
and no active rating run covers the latest pool
```
