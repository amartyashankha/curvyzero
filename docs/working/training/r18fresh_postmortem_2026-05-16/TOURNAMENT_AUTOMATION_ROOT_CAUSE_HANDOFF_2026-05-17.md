# Tournament Automation Root Cause Handoff - 2026-05-17

This doc is for the next agent who needs to answer: "why does this need manual
monitoring, and how do we make it self-healing?"

## Current Verified State

20:31 EDT:

- Full trainer batch status is fresh:
  - `136 / 136` rows accounted for;
  - `131` running, `5` completed, `0` failed;
  - checkpoint artifact sum `5510`;
  - latest iteration max `320000`.
- Manual refresh succeeded in catching trainer-opponent export up to the latest
  completed tournament rating:
  - latest rating `round-000040`, `4192` checkpoint refs;
  - trainer export generation `2` from `round-000040`;
  - export now reports current with latest rating.
- Remaining live blocker:
  - active game batch `round-000041`;
  - same `4192` checkpoint pool as `round-000040`;
  - queue length `0`;
  - new checkpoints not in latest rating `0`;
  - status still reports zero root-completed games.
- Concrete code bug found:
  - `spawn_if_existing` was used as the default for `spawn_if_empty`;
  - therefore a normal live continuation could start an empty same-pool rerate;
  - this explains repeated batches over the same `4192` refs and the need for
    manual intervention.
- Local fix and tests:
  - `spawn_if_empty = bool(payload.get("spawn_if_empty", False))`;
  - `spawn_if_existing` no longer implies empty same-pool rerating;
  - explicit `spawn_if_empty=True` still allows that behavior when deliberately
    requested;
  - focused pytest coverage passed.

20:18 EDT:

- Current status is better than the earlier stale read:
  - latest rating is now `round-000040` over `4192` checkpoint refs;
  - active game batch is `round-000041`, also over `4192` checkpoint refs;
  - fresh game output exists for `round-000041`.
- The trainer batch is alive: latest fast status saw `136` running and `0`
  failed rows.
- The remaining automation question is narrower:
  - trainer fast status reports `5217` checkpoint artifacts;
  - tournament intake reports `4192` checkpoint refs;
  - because run-status counts mirrored checkpoint artifacts and tournament
    intake only scans `attempts/<attempt>/train/lightzero_exp*/ckpt`, this may
    be a counting mismatch rather than a subscriber failure.
- Action before changing code: run a direct read-only tournament discovery over
  the manifest's `136` run IDs with `checkpoint_selection=all` and compare its
  found count to `4192`.
- Trainer export consumption is still unproven for generation `30`; do not call
  the loop fully closed until trainers load a latest export generated from the
  current latest rating.

20:00 EDT:

- Trainer app was redeployed and the saved `cz26-full-20260517a` manifest was
  relaunched.
- Relaunch produced `136` spawned train calls and `136` spawned poller calls.
- Fast trainer status after relaunch saw `111` running, `22` completed, `3`
  failed, and `5066` checkpoint artifacts.
- Latest tournament rating/export is still `round-000038` / generation `30`.
- Current missing proof: trainer-proof has not yet shown generation `30`
  assignment SHAs as latest-applied. The target count is currently `0`.
- This is the next boundary to monitor before claiming full automation.

19:53 EDT:

- Latest published rating is now `round-000038`, covering `4192` checkpoints.
- Trainer-facing export is generation `29`, sourced from `round-000038`, with
  `100` active rows and `24` rewritten pointers.
- Intake and latest rating both report `4192` checkpoints, so the current
  tournament pool is caught up.
- A same-pool rerate batch, `round-000040`, is active: `4192` checkpoints,
  `300` pairs, `6300` games, currently `828` root-completed games.
- The 136-run trainer batch is not running. The trainer app was stopped for
  capacity earlier and is absent from the current deployed app list.
- Therefore the tournament/export path recovered, but the end-to-end automated
  feedback loop is not alive until trainers are relaunched and prove they load
  generation `29` or later.
- Manual recovery remains necessary because the controller still lacks complete
  materialization/ownership records and the normal recovery path has timed out
  on large scans before. The correct permanent fix remains: compact result
  sidecars, lifecycle ledger, owner call ids, and safe self-healing skip/retry.

15:00 EDT:

- Active useful batch: `round-000038`.
- `round-000038` checkpoint count: `4192`.
- `round-000038` pair/game count: `300` pairs / `6300` games.
- Liveness probe saw real game output.
- Latest published rating is still `round-000036` until `round-000038` reduces.
- The failed/stale drain before this was
  `fc-01KRVMGPC5EZ8THYHQXPDV0CP4`; its child rating loop terminated after the
  app was stopped.
- The currently useful drain is `fc-01KRVMRCZ6JYSG28GYTFTR5C18`.

- Main tournament: `cz26-live-20260517a`.
- Rating run: `elo-cz26-live-20260517a`.
- Trainer-facing export:
  `cz26-live-20260517a-elo-cz26-live-20260517a-training`.
- Latest published rating: `round-000036`, `3427` checkpoints.
- Latest trainer export: generation `20`, sourced from `round-000036`.
- Intake count after that publish: `4192` checkpoints.
- Backlog after that publish: `765` checkpoints.
- Current fresh drain call to watch:
  `fc-01KRVMGPC5EZ8THYHQXPDV0CP4`.
- Training app was stopped for capacity, so trainer consumption of generation
  `20` is not yet proven.

## What Actually Went Wrong

1. The tournament state lives in named Modal Volumes/Dicts. Deploying code does
   not rewrite those records.
2. Live Modal workers can keep running after durable state says a batch was
   skipped, unless the app/calls are explicitly stopped.
3. The reducer originally tried to parse thousands of huge game summaries. Each
   summary included a large `action_trace`, so reducing roughly `6300` games
   meant reading gigabytes of JSON.
4. Status/control paths were easier to schedule than reducer paths, so the
   operator could see state but not publish it.
5. Active batch state did not carry enough ownership metadata. We often knew a
   batch existed, but not which Modal call owned it or whether it still had live
   children.

## Immediate Fixes Already Applied

- Bounded scheduling is active: `adaptive_v0`, `pairs_per_round=300`.
- The compact reducer path can extract only score/player fields from bloated
  summaries instead of loading full `action_trace` JSON.
- `round-000036` was reduced and published through that compact path.
- Trainer export generation `20` was refreshed from `round-000036`.
- Game warm buffers were set to zero so deploys do not reserve hundreds of idle
  game workers before control/reduce work can start.
- A fresh bounded drain was spawned for the newer checkpoint pool.

## Five Permanent Fixes To Evaluate

1. Write a compact `result.json` or pair tally sidecar when each game finishes.
   Reducers should never need to open full GIF/debug/action-trace summaries.

2. Add a checkpoint lifecycle ledger:
   `written -> discovered -> accepted -> scheduled -> game_output -> rated -> exported -> trainer_loaded`.
   This turns "are checkpoints making it?" into one query.

3. Add batch ownership records:
   drain call id, rating call id, game map call id, reducer call id, app id,
   input pool hash, pair count, game count, created time, and last output time.

4. Make stale/zero-output batch handling explicit and bounded:
   if a batch has no output after a short grace period, quarantine it and move
   on; if it has partial output, reduce what exists after a defined deadline.

5. Separate control/reduce from game fanout operationally. Game workers can be
   massive and disposable; publish/export must have a small reliable lane that
   cannot be starved by game batches.

## Validation Gates

- Fresh drain creates exactly one bounded active batch.
- The active batch covers current intake, not an old already-rated pool.
- Games write output.
- Rating publishes without parsing huge summaries.
- Trainer-facing export advances to that rating.
- Trainers load the new export.
- New checkpoints arriving during the batch are visible as backlog for the next
  batch, not missing.

## Parallel Investigation Threads

Agents have been asked to investigate:

- the state machine and durable/live split;
- Modal deployment/runtime semantics;
- checkpoint lifecycle and missing-checkpoint failure modes;
- skeptical validation gates and what could still be false.

Their findings should be merged back here or into
`TOURNAMENT_STABILITY_HANDOFF_2026-05-17.md`.

## First Wave Findings

- The same root cause appears from all angles: live compute and durable state
  are not tied together tightly enough.
- Redeploying code is not state repair. It does not rewrite existing manifests,
  rating inputs, latest pointers, claims, or accepted function calls.
- Queue length is not enough. We had `queue_len=0` while `765` checkpoints were
  newer than the latest rating. The real measure is lifecycle stage per
  checkpoint.
- A batch input is immutable after it is written. New checkpoints that arrive
  during a batch should become backlog for the next batch, not silently vanish.
- A skipped durable batch does not prove old compute stopped. We saw an app with
  many tasks while status had no useful active batch.
- The reducer must not read full debug summaries. The emergency compact parser
  worked, but the permanent fix is a compact game result/tally written at game
  completion.
- Export consistency is now proven for `round-000036`, but trainer consumption
  of export generation `20` is not proven because trainers were stopped.

## Minimal Permanent Patch Order

1. Persist owner call ids for every active batch:
   drain call, rating loop call, rating round call, reducer call, and app id.

2. Add a lifecycle ledger/index keyed by checkpoint ref:
   `accepted`, `scheduled_round`, `game_output_seen`, `rated_round`,
   `export_generation`, `trainer_loaded_generation`, and terminal/drop reason.

3. Add a compact per-game result sidecar or per-pair tally sidecar. Reducers
   should read that, not large action-trace summaries.

4. Make the drain pre-spawn invariant hard:
   desired pool hash/count must match refreshed intake, pair count must be
   bounded, and old active work only blocks if it covers the current desired
   pool and has a live owner.

5. Add explicit quarantine/skip states for zero-output, wrong-pool, old-pool,
   and oversized batches. These states must unblock newer checkpoint pools and
   show up in the website/control status.

6. Add a safe cleanup command that can show live owners and stop only known
   obsolete tournament calls/apps, with a dry-run first.

## Operator Vocabulary To Use

- `queued`: intake event exists but has not been drained.
- `drained_not_scheduled`: intake event was consumed but no active/latest round
  contains the checkpoint.
- `scheduled_not_rated`: checkpoint is in an active round input but has not yet
  reached latest ratings.
- `rated_not_exported`: checkpoint is in latest ratings but not trainer-facing
  export.
- `exported_not_loaded`: checkpoint is in trainer-facing export but trainer
  proof has not shown a provider load.
- `complete`: trainer proof shows the expected assignment generation loaded.
- `quarantined` or `dropped`: terminal state with an explicit reason.

Current example:

- `4192` checkpoints are in intake.
- `3427` are in latest rating `round-000036`.
- `4192` are scheduled in active `round-000038`.
- Therefore the `765` newer checkpoints are not missing and not queued; they
  are `scheduled_not_rated`.

## Important Interpretation Rules

- `queue_len=0` does not mean caught up. It can mean the refs moved from queue
  into a scheduled/running batch.
- App task count is not tournament truth. It is a pressure signal. It can
  include pending map children, retries, old accepted calls, and terminating
  leftovers.
- "No active batch" must mean both:
  1. no valid active Volume/Dict batch, and
  2. no live owner FunctionCall for the current or superseded generation.
- Stop/redeploy is a manual emergency reset, not proof of automation.

## Current Validation Gates

1. `round-000038` must publish:
   latest rating advances to `round-000038`, checkpoint count `4192`, and
   reduction metadata shows completed/partial coverage clearly.

2. Trainer export must refresh:
   generation must advance beyond `20`, source round must be `round-000038`,
   and pointer rewrite count/assignment SHAs must be present.

3. Trainer consumption must be proven after trainer relaunch:
   target assignment SHAs must match the export, provider false rows must be
   zero, and latest-applied counts must be measured against live trainer count.

4. Backlog must stay visible:
   new checkpoints arriving during `round-000038` should be shown as either
   scheduled in the active batch or backlog for the next batch.

5. Automation must be proven without app stop/redeploy:
   one later checkpoint must cross intake, scheduling, rating, export, and
   trainer load without manual repair.

## Exact Next Code Fix

The latest failure class is now precise:

- drain got a rating-loop call id;
- no durable active round existed yet;
- status could only see "no active batch";
- app task count was high, so live Modal work existed but was not tied to a
  domain batch.

The fix is a materialization barrier:

1. Before spawning `curvytron_rating_loop`, write a spawn-intent record with:
   desired pool hash, desired checkpoint count, expected round id/index if
   known, drain call id, created time, and state `spawn_requested`.

2. After spawn, store the rating-loop call id separately from the drain call id.
   Do not overload a single `function_call_id` field.

3. Status should classify the spawn:
   - `rating_spawn_pending_no_artifact`: call exists, no round input yet;
   - `rating_spawn_materialized`: expected round input exists and matches
     desired count/hash;
   - `rating_spawn_dead_no_artifact`: call failed/terminated/stale and no round
     input exists.

4. A drain request should only block future drains while it is materialized or
   live-pending. If it is `dead_no_artifact`, the next control tick should
   clear/replace the lease without `--ignore-drain-request-lease`.

5. When the rating loop actually starts a rating round, persist the rating-round
   call id before waiting on it.

Plain English: "spawned" is not enough. The system should only consider the
tournament batch alive after the round input artifact exists or a live child
call is explicitly tracked.
