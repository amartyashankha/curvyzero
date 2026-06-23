# What The Hell Is Going On - Handoff - 2026-05-17

This is the plain-language handoff for the current cz26 trainer/tournament
feedback loop. Start here before reading older docs.

## Current Answer

Update at 21:03 EDT:

- After the local computer restart, the live Modal state is still intact.
  This is not being driven by a local terminal process.
- Yes, the full trainer batch is running/completing:
  `136 / 136` rows accounted for, `92` completed, `44` running, `0` failed.
  Checkpoint artifact sum is `7746`.
- The tournament is currently doing useful bounded work, not the old giant
  all-pairs job:
  - active internal game batch: `round-000043`;
  - `4512` checkpoint refs;
  - `300` pairs / `6300` games;
  - no blockers in the latest status probe;
  - game output is landing;
  - call graph sample shows `74` tournament game workers succeeded and `22`
    still pending.
- Latest completed rating is still `round-000040` over `4192` checkpoint refs.
  That means the newest active batch has not yet published ratings.
- Trainer feedback for the current export is partially proven and improving:
  - current trainer export still comes from `round-000040`, generation `2`;
  - `46` latest-applied target assignments across `136` runs;
  - latest decisions: `37` applied, `99` unchanged;
  - provider-false rows: `0`.
- `17` newer checkpoints arrived after `round-000043` started. They are queued
  for the next bounded batch and should not cause this active batch to be
  skipped.
- Current honest stability answer: stable through checkpoint discovery,
  intake, bounded tournament game execution, and current-export trainer
  consumption. Not yet fully stable through the newest publish/export/consume
  cycle until `round-000043` finishes and the trainers consume its export.

Update at 20:31 EDT:

- Yes, the full trainer batch is running/completing:
  `136 / 136` rows accounted for, `131` running, `5` completed, `0` failed.
- The tournament is alive but wasting work:
  - latest useful rating is `round-000040`, `4192` refs;
  - current active game batch `round-000041` repeats that same pool;
  - queue length is `0`, so this batch is not processing new checkpoints.
- Manual refresh caught trainer-opponent export up to the latest useful rating:
  generation `2` from `round-000040`.
- Concrete bug found and patched locally:
  `spawn_if_existing` accidentally meant `spawn_if_empty`. In plain English:
  the code was allowed to start another tournament game batch even when there
  were no new checkpoint events. That explains the repeated same-pool batches.
- Focused tests for that guard pass. The patch is not deployed yet.
- Remaining proof gap: after deploy/repair, prove trainers actually load the
  refreshed assignment and prove the next fresh checkpoint intake creates one
  bounded useful tournament batch.

Update at 20:36 EDT:

- The patch is deployed now.
- Stopping the old tournament app did not delete checkpoint/rating state; those
  are in persistent Volumes/Dicts.
- The old active game batch still exists in durable state. Recovery scan found
  it is not dead:
  - `1431 / 6300` games found complete;
  - `105 / 300` pairs started;
  - latest output was recent enough, so recovery correctly refused to skip it.
- This active batch is still not useful strategically because it repeats the
  same `4192` refs already rated in `round-000040`, but it should be the last
  accidental same-pool rerate caused by the `spawn_if_existing` bug.
- Trainer feedback is partially proven:
  - export from `round-000040` is current;
  - `32` trainers have already applied latest target assignment SHAs;
  - `0` provider-false rows.
- Next proof: wait for the current active batch or recovery state to clear,
  then verify the next newly discovered checkpoints produce one bounded batch
  and that trainers later consume the next export without manual refresh.

Update at 20:42 EDT:

- That next proof moved forward:
  - intake discovered `270` new checkpoint refs;
  - intake pool advanced to `4462`;
  - the stale old same-pool batch was skipped;
  - a new bounded rating call was spawned:
    `fc-01KRW8GE5GRD2M3J0TF6G9YWYB`.
- Full trainer batch is still healthy:
  `82` running, `54` completed, `0` failed.
- What remains unproven:
  - the new rating call has not yet visibly produced a new active game batch or
    new ratings in status;
  - after it completes, trainer export must advance past generation `2`, and
    trainers must consume that new generation.

Update at 20:44 EDT:

- The new rating call did produce a new active game batch:
  - `round-000042`;
  - `4462` checkpoint refs;
  - `300` pairs / `6300` games;
  - covers current intake;
  - is newer than latest rating;
  - queue length is now `0`.
- This is the clean automation behavior we wanted after the patch.
- Still not complete: wait for game output, ratings written, export refresh,
  and trainer consumption of the new export.

Update at 20:51 EDT:

- New issue found: `round-000042` was skipped too early after more checkpoints
  arrived. That was another recovery bug, not a trainer failure.
- Correct rule: a batch that already started should keep running against the
  pool it was launched with. Newer checkpoints become the next batch. They
  should not invalidate the current batch.
- Patch deployed:
  - do not skip a zero-output different-spec batch unless it is stale or no
    newer than the latest completed rating;
  - tests passed.
- New patched drain spawned rating call:
  `fc-01KRW9208TSJDJ8S7NR0BAF6G9`, using `4512` checkpoint refs.
- Next proof: this patched call should create a bounded game batch and keep it
  alive while additional checkpoints arrive.

Update at 20:53 EDT:

- The patched call created `round-000043`.
- `round-000043` is the current active batch:
  - `4512` checkpoint refs;
  - `300` pairs / `6300` games;
  - queue `0`;
  - covers current intake;
  - newer than latest rating.
- It was only `29s` old at the check, so no game output yet is not a problem.
- Next proof: wait/recheck that it remains alive and starts writing game output.

Update at 20:18 EDT:

- The `136`-row trainer batch is running again. Latest fast status saw all
  `136` rows running, no failed rows, and `5217` trainer-visible checkpoint
  artifacts.
- The tournament is not dead. Latest rating advanced to `round-000040`, and
  `round-000041` is currently playing with fresh game output.
- The closed loop is still not fully proven. Tournament intake/latest rating
  still report `4192` checkpoint refs, while trainer fast status reports
  `5217` artifacts. This may be a real intake miss, or it may be duplicate
  artifact accounting because run-status counts mirrored checkpoint artifacts
  that tournament intake does not use.
- The next proof is explicit: compare tournament discovery over the `136`
  manifest run IDs against the active manifest count. If discovery sees more
  than `4192`, fix subscriber/manifest state. If it sees `4192`, update status
  language and stop treating trainer artifact sum as tournament backlog.
- Trainer assignment consumption remains pending: export generation `30` is
  still not latest-applied by the `136` trainers.

Update at 20:00 EDT:

- The full 136-row trainer batch has now been relaunched from the saved
  `cz26-full-20260517a` manifest.
- Relaunch result:
  - `136 / 136` train calls spawned;
  - `136 / 136` poller calls spawned;
  - `24` assignment artifacts and `24` refresh pointers written;
  - trainer app is deployed with active tasks.
- Fast status says:
  - `111` running;
  - `22` completed;
  - `3` failed;
  - all `136` have heartbeat/progress files.
- Tournament/export is current through `round-000038`, export generation `30`.
- The remaining proof gap is trainer consumption of generation `30`:
  `trainer-proof` currently reports latest-applied target count `0`.
  Do not call the loop closed until this changes.

Update at 19:53 EDT:

- The 136-run `cz26-full-20260517a` trainer batch is not running right now.
- The trainer app was stopped earlier for capacity and has not been relaunched.
- The tournament/export side did recover:
  - latest rating is `round-000038`;
  - latest rating covers `4192` checkpoints;
  - trainer-facing export is generation `29` from `round-000038`.
- Current intake and latest rating both have `4192` checkpoints, so there is no
  missing-checkpoint backlog in the current tournament pool.
- A same-pool bounded rerate batch, `round-000040`, is active with `4192`
  checkpoints, `300` pairs, and `6300` games. It has game output but the latest
  sampled output is stale, so recovery handling still needs work.
- The full loop is not currently closed because no live trainers are producing
  new checkpoints or proving they loaded export generation `29`.
- The reason this is still manual is not mysterious anymore: durable tournament
  state, live Modal workers, and trainer relaunch state are not owned by one
  self-healing controller with enough persisted ownership/lifecycle records.

Update at 15:19 EDT:

- `round-000038` is still running, still current, and still producing fresh
  sampled output.
- Latest rating/export have not moved past `round-000036` / generation `20`.
- The correct action remains wait/recheck, not duplicate drain.

Update at 15:08 EDT:

- `round-000038` is still running and still the right batch.
- It covers `4192` checkpoints and has liveness output.
- A sampled call graph shows game children succeeding while the rating
  loop/round remains pending.
- Latest rating/export have not moved past `round-000036` / generation `20`
  yet.
- The correct action is wait/recheck, not duplicate drain.

Update at 15:00 EDT:

- A fresh useful bounded batch is now running: `round-000038`.
- It covers the current `4192` checkpoint intake pool.
- It is bounded: `300` pairs / `6300` games.
- A liveness probe saw real game output. This is the first proof after the
  latest stop/redeploy that game fanout is alive again.
- The latest published rating is still `round-000036` until `round-000038`
  reduces.
- The trainer export is current with `round-000036`, generation `20`.
- The live path is better, but not stable enough to relaunch trainers yet.

Update at 14:52 EDT:

- The immediate stuck publish was fixed.
- `round-000036` reduced and published as the latest rating:
  - checkpoint count: `3427`;
  - game count: `6300`;
  - pair count: `300`.
- The trainer-facing export was refreshed from `round-000036`:
  - generation: `20`;
  - active rows: `100`;
  - rewritten trainer assignment pointers: `24`.
- Intake now has `4192` checkpoints, so `765` checkpoints are newer than the
  latest published rating.
- `round-000037` was attempted for the `4192` pool, but it had zero output and
  was skipped. That means it is not currently blocking the pipeline, but it is
  evidence that the live compute/ownership path is still fragile.
- A fresh bounded drain was spawned at 14:52 EDT:
  `fc-01KRVMGPC5EZ8THYHQXPDV0CP4`.
- The training app was stopped earlier to free capacity, so do not claim
  trainer consumption of generation `20` until trainers are relaunched or a
  trainer-proof artifact proves that generation was loaded.
- The deeper issue is not "old tournaments." The deeper issue is that durable
  state and live Modal workers can disagree. We need explicit ownership records,
  checkpoint lifecycle records, and a guard that prevents stale or zero-output
  batches from blocking future checkpoint pools.

Plain English: the system moved past the immediate stuck batch. It is not yet a
self-healing automated loop. The next proof is: fresh drain creates a bounded
batch, games produce output, rating publishes, export refreshes, and trainers
load that export without manual intervention.

The system is not fully stable yet. It is also not totally broken.

Update at 13:55 EDT:

- The current active game batch is `round-000036`.
- It is the right batch: `3427` checkpoints, bounded to `300` pairs and
  `6300` games.
- It has real game output. A progress pass found `6289 / 6300` games complete
  and `299 / 300` pairs complete.
- The latest published rating is still `round-000034` / `2739` checkpoints.
- Intake has advanced to about `4107` checkpoints, so roughly `680`
  checkpoints are backlog for the next batch after `round-000036`.
- The immediate blocker is publish/reduce. The normal path waits for the last
  `11` games; manual partial reduce attempts are not getting scheduled.
- Concrete repair attempts already made:
  - added `curvytron_rating_reduce_rescue`;
  - lowered reducer resources to `0.25 CPU / 1GB`;
  - added `curvytron_feedback_loop_reduce_rescue` using the same volume set as
    the status/control function;
  - reduced tournament game warm buffers to zero;
  - stopped stale tournament apps and redeployed current code several times;
  - cancelled obsolete pending reducer calls.
- Current active rescue call:
  `fc-01KRVH57NGQY1VZDSVC8N65886`.
- The key observed problem: status can run, but reducer/rescue calls stay
  `PENDING` with no task id. This is not a reducer exception yet; it is a
  scheduling/control-lane problem.
- Capacity action at 14:00 EDT: stopped training app
  `ap-FTBsuB0JXLZoA5MYhadNYv` because it was holding `100+` tasks and the
  tournament subscriber/reducer could not get reliable CPU. Committed
  checkpoints remain on the v2 volumes; running trainers must be relaunched
  after publish is proven.
- After stopping trainers, the tournament app was stopped/redeployed again.
  Current tournament app id: `ap-RTOCoNbF6UIYfSNotvgvBS`.
- Current live attempt: embedded `mode=reduce_rescue` through
  `curvytron_feedback_loop_status`.
- The first embedded attempt proved the function actually starts, but it hit
  the old 300-second status timeout. Current code raises that timeout to
  30 minutes and the embedded rescue is running again.

The trainer side is mostly healthy: the launched training jobs are writing many
checkpoints, and trainers have proven that they can load the current
trainer-facing leaderboard export.

The tournament side is the slow/brittle part. Two tournament game batches have
now published, including the oversized `round-000034`. The latest immediate
repair was:

- `round-000035` was a bounded but wrong-pool batch. It only contained the same
  `2739` checkpoints already rated by `round-000034`, while intake had `3427`.
- State marked `round-000035` skipped, but Modal logs showed its old game
  workers were still running and timing out. So state repair alone was not
  enough; the bad worker swarm could still consume capacity.
- We stopped the deployed tournament app, redeployed current code, and spawned
  one fresh detached drain for the `3427`-checkpoint pool.

The current proof gap is now: confirm the fresh drain creates a new bounded
game batch that actually covers all `3427` intake checkpoints, then confirm it
publishes and trainers consume the export.

## Live Names

- Main launch manifest: `cz26-full-20260517a`
- Tournament id: `cz26-live-20260517a`
- Rating run id: `elo-cz26-live-20260517a`
- Trainer-facing leaderboard/export:
  `cz26-live-20260517a-elo-cz26-live-20260517a-training`
- Modal app: `curvyzero-checkpoint-tournament-v2`
- Tournament website:
  `https://modal-labs-shankha-dev--curvyzero-checkpoint-tournament--93d419.modal.run/`
- Current spawned next drain call: `fc-01KRV9TXJHWSVXF2ZY046RQP4F`
- Current reducer rescue call: `fc-01KRVH57NGQY1VZDSVC8N65886`
- Latest published tournament rating: `round-000034`

Important naming note: `round-000034` is only an internal tournament game-batch
directory. It is not a training round. Trainers keep running continuously.

## What Has Been Proven

- Trainers wrote checkpoints.
- Intake saw those checkpoints.
- Tournament game-batch `round-000033` rated `2192` checkpoints.
- `round-000033` completed `22575` games and published ratings.
- Trainer-facing export generation `11+` was written from `round-000033`.
- Running trainers loaded assignments from that export.
- Provider-load proof had many `provider-ok` rows and `0 provider-false` rows.

Plain English: the full data path has happened at least once.

## What Is Not Proven Yet

- Newer checkpoints after `round-000034` are still waiting for the next
  tournament result.
- `round-000036` has not yet published a rating.
- A trainer export sourced from `round-000036` has not yet been written.
- Trainers have not yet proven they loaded a `round-000036` export.
- We have not yet seen trainers load a trainer-facing export sourced from
  `round-000034` in the latest proof pass.
- The next batch was just spawned; verify that it uses the new hard cap and does
  not become another `1348`-pair batch.

Plain English: the tournament moved forward; trainer refresh/proof is now the
next boundary to validate.

## Current Live State At 11:23 EDT

Status immediately before recovery said:

- intake checkpoints: `3427`
- latest published rating checkpoint count: `2192`
- checkpoints newer than latest rating: `1235`
- checkpoint queue length: `622`
- active tournament game-batch: `round-000034`
- `round-000034` checkpoint count: `2739`
- `round-000034` pairs: `1348`
- `round-000034` games: `28308`
- visible root completed games: `4414 / 28308`
- latest trainer-facing export: generation `15`, still sourced from
  `round-000033`

The cheap liveness sample is stale, so the correct next action is not guessing
from old docs or the website. The correct next action is a recovery scan through
the control tool.

Recovery immediately after that found `round-000034` complete:

- latest published rating: `round-000034`
- latest rating checkpoint count: `2739`
- `round-000034` completed games: `28308 / 28308`
- checkpoints newer than latest rating: `688`
- queue length after repair/drain: `0`
- new rating call spawned: `fc-01KRV8JHYV02SJ8GNH1SJPGQW4`
- trainer-facing export was still stale at that moment, generation `15` from
  `round-000033`

Follow-up at about 11:27 EDT:

- `refresh-if-ready` wrote trainer-facing generation `16` from `round-000034`.
- The next active batch is `round-000035`.
- `round-000035` is bounded correctly: `300` pairs and `6300` games.
- But `round-000035/input.json` contains exactly the old latest pool:
  `2739` refs, not the current intake pool of `3427`.
- The 688 newer checkpoint refs are in the intake manifest but not in
  `round-000035`.

Plain English: the size cap fix worked, but the next batch is still not the
catch-up batch we need. It is re-rating the already-published pool, so it should
not block newer checkpoints.

## Why This Happened

There were two real bugs.

First, live Modal state persists. Patching defaults in code did not rewrite an
already-created active tournament game batch. That means a bad live batch can
keep running after the code is fixed.

Second, the adaptive tournament scheduler did not treat `pairs_per_round=300`
as a hard cap. It expanded the active batch to `1348` pairs, which became
`28308` games. That is why the current batch is slow.

There was also a control-plane bug: a stale active batch could be blocked as
"already active" before the recovery logic had a chance to scan real game
outputs.

## Fixes Already Applied

- Future adaptive tournament batches now respect `pairs_per_round` as a hard
  cap.
- Stale active batches are allowed to enter recovery scan.
- Active batches that only cover the already-published pool while newer
  checkpoints are waiting are now allowed to enter recovery scan too.
- The skip decision now has an explicit reason:
  `different_spec_already_rated_pool`.
- Scheduled drain ticks use a small activity probe so stale active output is
  visible.
- Large-batch recovery uses bounded activity probing instead of scanning tens of
  thousands of game summaries.
- Old active batches can now be marked due for partial reduce after real output
  exists for more than one hour.
- `curvytron_checkpoint_intake_drain` can spawn
  `curvytron_rating_reduce(..., allow_partial=True)` for an old incomplete
  active batch instead of waiting for every straggler game.
- Operator status/control now flags `active_game_batch_partial_reduce_due`.
- Operator output includes the recovery decision so we can see why a batch was
  skipped or left alive.
- Focused test slices passed for adaptive scheduling, control decisions, and
  recovery behavior.
- The tournament app was redeployed with those fixes.

## 13:12 EDT Current Blocker And Repair

Plain English:

- `round-000036` is correct, bounded, and alive.
- The old synchronous recovery command was too slow and was killed locally after
  the direct reduce path was spawned.
- The first direct detached partial reduce stayed pending in an ephemeral
  local-entrypoint app, so that app was stopped.
- The deployed tournament app was also stopped to kill the live game swarm,
  then redeployed.
- Current deployed-app partial reduce for `round-000036`:
  `fc-01KRVF11V4F7NVABHYDAWYY8S9`.
- Follow-up finding: game workers had enormous warm-container settings
  (`100/400` for single games and `500/500` for shards). That can reserve
  hundreds of idle game workers immediately after deploy and starve the
  control/reduce lane. Current code lowers both game worker warm settings to
  `min=0`, `buffer=16`.
- Follow-up finding: `curvytron_rating_reduce` was asking for `2 CPU / 8GB`.
  Status functions with `1GB` could run while reduce stayed pending with no task
  id, so current code lowers reduce to `1 CPU / 2GB` to make it schedulable.
- After cancelling obsolete reduce calls, the current reduce call to watch is
  `fc-01KRVFHF8025JT906B4M7VGXXP`.
- Status still reports latest rating `round-000034` until that reduce writes
  ratings.
- Once it publishes, immediately run `refresh-if-ready`, then
  `trainer-proof`, then a bounded `drain-if-ready` for the backlog.

Evidence:

- Status flags include `active_game_batch_partial_reduce_due`.
- Active batch: `round-000036`, `3427` checkpoints, `300` pairs, `6300` games.
- Intake: about `3956` checkpoints; latest rating: `2739`; backlog after the
  active batch: about `529`.
- Modal logs show fresh `round-000036` game summaries and some very long games.
  That explains why waiting for all games can stall publication.

## Root-Cause Exploration Started

Parallel reviewers found the same broad issue: Modal durable state and Modal
live compute are different things.

Five concrete deeper fixes to consider:

1. Drain should refresh checkpoint discovery before it builds the rating pool,
   so it cannot spawn from a stale manifest.
2. Rating spawn should enforce a pool invariant: if intake has newer
   checkpoints and the desired batch does not include them, refuse to spawn and
   say `manifest_refresh_required`.
3. Each active batch should persist its owner/call IDs: drain call, rating-loop
   call, rating-round call, reduce call, and a compact call-graph snapshot.
4. Status should maintain a cheap progress index: expected games, completed
   games, failed games, latest game output time, and count basis.
5. Add per-checkpoint lineage events:
   `discovered -> accepted -> queued -> drained -> scheduled -> rated -> exported -> trainer_loaded`.

The deepest current problem is not that we need to watch harder. The deeper
problem is that the system does not yet have enough first-class state to answer
"where is this checkpoint stuck?" or "who owns this active batch?" directly.

## Current Recovery Command

Use this when status reports stale active output:

```bash
uv run --extra modal python scripts/curvytron_live_loop_control.py --action drain-if-ready --activity-probe-pairs 8 --lookahead-batches 64 --max-events 2000 --no-drain-call-probe
```

Expected behavior:

- If the active batch has fresh real game output, it should not be skipped.
- If the active batch is truly stale/dead, recovery should skip or move past it
  without blocking the whole pipeline.
- It should not spawn a duplicate drain just because a game batch is already
  active.

## 11:46 EDT Immediate Repair

Plain English:

- Redeploying code does not erase named Modal Volumes or Dicts, so it should not
  lose checkpoints, ratings, or intake manifests.
- Stopping the deployed app does kill currently running function calls, including
  orphaned workers from skipped game batches.
- That was necessary because `round-000035` was skipped in state but old game
  workers were still running in Modal logs.
- After redeploy, status showed no active game batch, latest rating
  `round-000034` / `2739`, intake `3427`, and `688` checkpoints not yet rated.
- A new detached drain was spawned:
  `fc-01KRV9TXJHWSVXF2ZY046RQP4F`.

Next check:

1. Done: status shows new active game batch `round-000036`.
2. Done: `round-000036` contains `3427` checkpoints, not `2739`.
3. Done: it is bounded at `300` pairs and `6300` games.
4. Partly done: game output is landing for `round-000036`, and the compact call
   graph shows visible game children as `SUCCESS`; wait for the rating round to
   reduce and publish. The slow full progress probe still times out at `300s`.
   A detached recovery scan was spawned as `fc-01KRVB3CVZH2VHH7R0EJS0ZST5`
   because the liveness sample went stale.
5. Partly done for the current latest rating: `trainer-proof` for generation
   `17` from `round-000034` found `39218` provider-ok rows, `0`
   provider-false rows, and `48 / 136` runs with a target assignment as their
   latest applied assignment. After `round-000036` publishes, repeat
   `refresh-if-ready` and `trainer-proof` for that new rating.

Do not call this stable until those five checks pass.

## What To Check Next

1. Confirm the fresh drain `fc-01KRV9TXJHWSVXF2ZY046RQP4F` materializes a
   bounded active game batch for the `3427`-checkpoint pool.
2. Confirm the old `round-000035` worker swarm is gone after app stop/redeploy.
3. Run `trainer-proof` for generation `17` from `round-000034`.
4. If the new batch again uses only `2739` checkpoints, fix the drain/rating
   handoff so the rating loop reloads the latest manifest immediately before it
   writes input.
5. Add permanent protection: skipped state must either cancel/abandon matching
   live workers or the app cleanup/runbook must explicitly do it.

## Remaining Deeper Fixes To Consider

- Add a cheap progress index so status can report total completed games without
  scanning many files.
- Add straggler replay or partial reduce so one slow game does not block the
  whole batch forever.
- Add a hard live guard: a live tournament should reject huge all-pairs or
  oversized adaptive batches before workers launch.
- Make duplicate scheduled calls return "already active" cleanly instead of
  crashing or spawning conflicting work.
- Add lineage events for every checkpoint boundary:
  checkpoint written, intake seen, tournament scheduled, game result written,
  rating published, trainer export written, trainer assignment loaded.

## Simple Stability Standard

Do not say "stable" because a website loads or a game batch exists.

Say "stable" only when:

- trainers are writing checkpoints;
- intake count increases;
- tournament batches are bounded and publish;
- trainer-facing export advances from the latest published rating;
- trainers load that export;
- status shows the next backlog is being picked up without manual babysitting.
