# cz26 Current Truth - 2026-05-17

Use this as the short operator note for the live cz26 loop.

## 2026-05-18 10:24 EDT Batch Rationale Reorientation

- Added `CZ26_BATCH_RATIONALE_REORIENTATION_2026-05-18.md`.
- Current plain read: the `136`-row `cz26-full-20260517a` batch was an
  attribution experiment after `r18fresh`, not a random grab bag.
- Grid A: `96` rows testing four mixed opponent recipes across reward alpha,
  action noise, and leaderboard-opponent immortality.
- Grid B: `40` rows testing opponent-slot populations directly, including pure
  blank, pure wall-avoidant, and pure rank1 controls.
- Keep reward progression, survival progression, and tournament performance as
  separate signals. Reward can fall when tournament-fed opponents become
  stronger; that is not automatically bad.

## 2026-05-18 10:20 EDT Tournament GIF Fix

- Root cause: the scale-probe setting `save_gif=false` leaked into the live
  tournament intake config. That meant tournament games still ran and ratings
  still updated, but battle rows had no GIF refs for the website.
- Current live Modal Dict was repaired from:
  - `save_gif=false`
  - `gif_sample_games_per_pair=0`
  to:
  - `save_gif=true`
  - `gif_sample_games_per_pair=5`
- Code now treats GIFs as required for live tournament intake. If a live intake
  manifest says GIFs are off, the repair path turns them back on and records
  `live_gif_repaired_to_enabled=true`.
- Current limitation: any game batch already created before this repair keeps
  its immutable input. For example, active batch `round-000052` was already
  created with `save_gif=false`, so it will not grow GIFs mid-flight. The next
  newly-created live batch should use the repaired `save_gif=true`,
  `gif_sample_games_per_pair=5` config.
- Scale probes may still use GIFs-off in separate one-off runs. The live
  feedback tournament should not.
- Focused proof: `tests/test_curvytron_checkpoint_intake_repair.py` now includes
  `test_live_intake_legacy_gifs_off_is_repaired` and
  `test_live_intake_put_repairs_gifs_off_before_persisting`.

## 21:01 EDT Current Truth

- After the local computer restart, Modal state is still intact; this is not a
  local-process-driven loop.
- The full `cz26-full-20260517a` trainer batch is still accounted for:
  - `136 / 136` rows;
  - `92` completed;
  - `44` running;
  - `0` failed;
  - checkpoint artifact sum `7746`;
  - checkpoint count range `11..70`;
  - latest iteration range `50000..326739`;
  - grid split remains `cz26a:96`, `cz26b:40`.
- Current tournament status remains bounded and useful:
  - active internal game batch: `round-000043`;
  - `4512` checkpoint refs;
  - `300` pairs / `6300` games;
  - queue length `0`;
  - `320` checkpoint refs newer than latest completed rating;
  - game output is landing and the liveness sample is fresh.
- Latest completed rating is still `round-000040` over `4192` checkpoint refs.
  Trainer export is still generation `2`, sourced from `round-000040`.
- Plain English: the loop is stable through checkpoint discovery, intake, and
  bounded tournament game execution. It is not yet proven complete for the
  newest batch until `round-000043` writes ratings, publishes latest, refreshes
  the trainer export, and trainers visibly consume that newer export.
- A fresh `trainer-proof` scan is running now to re-check trainer consumption
  of the current exported opponent list.

## 21:03 EDT Update

- Fresh `trainer-proof` completed:
  - `136` runs scanned;
  - latest-applied target count is now `46`;
  - latest refresh decisions: `37` applied, `99` unchanged;
  - provider rows: `46688` ok, `0` false, `55560` null/pending.
- Plain English: trainers are still reading valid tournament-exported opponent
  assignments. This proof is for the current trainer export, which still comes
  from `round-000040`.
- Fresh tournament status with liveness probe:
  - `round-000043` is still active;
  - `4512` checkpoint refs, `300` pairs / `6300` games;
  - no blockers;
  - game output is still landing;
  - call graph sample shows `74` tournament game workers succeeded and `22`
    still pending.
- `17` newer checkpoints arrived after `round-000043` started. They are queued
  for the next bounded batch; they should not invalidate this running batch.
- Remaining proof is unchanged: wait for `round-000043` to publish ratings,
  then verify trainer export advances past generation `2` and trainers consume
  that newer export.

## 21:13 EDT Trainer Length/Progress

- Intended training length from the launch manifest:
  - `max_train_iter=300000`;
  - checkpoints every `10000` learner iterations;
  - `max_env_step=30000000`, which is not the binding limit here;
  - background poller `max_runtime_sec=21600` (`6h`).
- Fresh fast-table status:
  - `136 / 136` rows accounted for;
  - `104` completed;
  - `32` running;
  - `0` failed;
  - checkpoint artifact sum `8062`;
  - checkpoint count range `11..70`;
  - learner iteration range `55069..326739`.
- Plain English: most runs have hit the intended `300k` learner-iteration
  target or overshot slightly while shutting down. The remaining active set is
  mostly close to done, with two real laggards:
  - `cz26b-r028-out50-n10-imm10-b20w05r1`: about `55k / 300k`;
  - `cz26a-r013-out67-n0-imm0-b20w05r1`: about `181k / 300k`.
- Expected finish:
  - most remaining runs should finish soon because many are already at
    `250k-305k`;
  - the true tail depends on whether the `55k` laggard keeps making progress or
    gets preempted/restarted. Do not call the whole batch complete until that
    row either reaches `300k` or is intentionally retired.

## 22:35 EDT Website Links/Smoke

- Current public links from Modal:
  - Tournament:
    `https://modal-labs-shankha-dev--curvyzero-checkpoint-tournament--93d419.modal.run/`
  - GIF browser:
    `https://modal-labs-shankha-dev--curvyzero-curvytron-gif-browser--f71ce8.modal.run/`
- Important correction: the older GIF browser URL ending in `--bada8e` is
  stopped and returns `404`.
- Smoke test results:
  - Tournament root: HTTP `200`, about `1.75 MB`, about `15.9s`.
  - Tournament `/api/tournaments?fresh=1`: HTTP `200`, shows
    `cz26-live-20260517a` as current and in `rating` status.
  - Tournament `/api/rating-standings?fresh=1&limit=20&live_provisional=1`:
    HTTP `200`, `4192` rows from latest completed rating `round-000040`.
  - GIF browser root: HTTP `200`, about `153 KB`, about `15.9s`.
  - GIF browser `/api/summaries?fresh=1&limit=5`: HTTP `200`, returns `5`
    rows, `18` current runs, and `reload_error=null`.
- Honest limitation: this is an HTTP/API smoke test, not a full browser visual
  regression. I have not done a Playwright screenshot/click pass in this check.

## 20:31 EDT Current Truth

- The full `cz26-full-20260517a` trainer batch is still real and alive.
- Fresh fast trainer status:
  - `136 / 136` heartbeats;
  - `136 / 136` progress files;
  - `131` running;
  - `5` completed;
  - `0` failed;
  - checkpoint artifact count range `9..69`;
  - total trainer-visible checkpoint artifacts `5510`;
  - latest iteration range `30000..320000`;
  - grid split remains `cz26a:96`, `cz26b:40`.
- The tournament has caught the trainer-opponent export up to the latest
  completed rating:
  - latest rating: `round-000040`, `4192` checkpoint refs;
  - trainer export: generation `2`, sourced from `round-000040`;
  - `trainer_export.current_with_latest_rating=true`.
- The remaining live tournament problem is not "old tournaments"; it is the
  current active game batch:
  - active game batch: `round-000041`;
  - same `4192` checkpoint refs as the latest completed rating;
  - queue length `0`;
  - new checkpoints not in latest rating `0`;
  - `300` pairs / `6300` games;
  - status still reports zero root-completed games.
- Root cause found in code: `spawn_if_existing` was being treated as
  `spawn_if_empty`. That allowed the live drain to start another same-pool
  tournament batch even when no new checkpoints were queued. This is why the
  loop kept needing manual judgement and why the tournament app accumulated
  lots of tasks.
- Local fix made and tested:
  - `spawn_if_empty` now only means explicit `spawn_if_empty`;
  - `spawn_if_existing` no longer starts empty same-pool rerates;
  - focused tests passed:
    `test_intake_drain_spawn_if_existing_does_not_force_empty_same_pool_continuation`,
    `test_intake_drain_spawn_if_empty_allows_explicit_same_pool_continuation`,
    and
    `test_checkpoint_intake_drain_tick_spawns_drain_without_local_mutation`.
- Next live action: deploy the patched tournament app and clear/skip the
  current same-pool waste batch if it is still blocking useful work.

## 20:36 EDT Update

- Patched `curvyzero-checkpoint-tournament-v2` was deployed.
- The previous tournament app instance was stopped; it is still draining old
  tasks according to Modal app list, but the new deployed app is live.
- A recovery drain was run against the old active game batch. It did **not**
  skip it because the full output scan found useful live data:
  - active game batch: `round-000041`;
  - completed games found by recovery scan: `1431 / 6300`;
  - started pairs found: `105 / 300`;
  - latest game output was recent enough for the recovery rule
    (`stale_age_seconds` about `137`, below the `600` second threshold).
- Plain English: `round-000041` is wasteful because it repeats the same pool as
  latest rating, but it is not dead. The important fix is deployed so the next
  control decision should not start another empty same-pool batch by accident.
- Trainer feedback proof improved:
  - trainer export is current with `round-000040`;
  - `trainer-proof` saw `32` trainers apply one of the latest target assignment
    SHAs;
  - `104` trainers were unchanged at the latest check;
  - `target_provider_false_count=0`.
- Plain English: the feedback loop is partially proven live: tournament rating
  has been exported and some trainers have loaded it. We still need to watch
  the next checkpoint/update cycle to prove the loop repeats automatically
  without manual nudging.

## 20:42 EDT Update

- After deployment, the next live intake cycle worked:
  - intake count advanced from `4192` to `4462`;
  - `270` checkpoint refs were newer than the latest rating;
  - the old same-pool batch `round-000041` was skipped with reason
    `different_spec_zero_output`;
  - a new rating call was spawned:
    `fc-01KRW8GE5GRD2M3J0TF6G9YWYB`;
  - desired rating pool is now `4462` checkpoint refs.
- This proves new trainer checkpoints are making it into tournament intake.
- Fresh trainer fast status after the quiet window:
  - `136 / 136` rows accounted for;
  - `82` running;
  - `54` completed;
  - `0` failed;
  - checkpoint artifact sum `6821`;
  - latest iteration max `320847`.
- Open proof: the new rating call is pending with children
  `curvytron_rating_loop`, `curvytron_rating_round`, and provisional loop.
  Next check must show a bounded game batch or ratings output.

## 20:44 EDT Update

- The new rating call materialized into a real bounded game batch:
  - new active game batch: `round-000042`;
  - checkpoint refs: `4462`;
  - queue length: `0`;
  - `new_checkpoints_not_in_latest_rating`: `270`;
  - pair count: `300`;
  - game count: `6300`;
  - active batch covers current intake: true;
  - active batch is newer than latest rating: true.
- This proves the automated leg:
  `trainer checkpoints -> intake discovery -> drain -> bounded tournament game batch`.
- The batch was only about `21s` old at the check, so no game output was
  expected yet. Next proof is game output, rating publish, export refresh, then
  trainer consumption of the new export.

## 20:51 EDT Update

- A second bug was found after `round-000042` started:
  - newer checkpoints arrived while `round-000042` was young;
  - recovery compared the running batch against the newer desired pool;
  - it skipped `round-000042` before it had time to produce output.
- This is wrong for live training. A running batch must be allowed to finish
  against the pool it was launched with; newer checkpoints should wait for the
  next bounded batch.
- Local code was patched and redeployed:
  - a zero-output different-spec batch is only skippable if it is actually
    stale or already no newer than the latest rating;
  - focused test proves a newer zero-output batch is kept until stale;
  - intake-repair suite still passes (`37 passed`);
  - skip/drain tournament subset passes (`8 passed`).
- New drain after the patch:
  - current intake: `4512` refs;
  - `320` refs newer than latest rating;
  - new rating call: `fc-01KRW9208TSJDJ8S7NR0BAF6G9`.
- Next proof: this new call must materialize a bounded game batch and not
  self-skip merely because more checkpoints arrive.

## 20:53 EDT Update

- The patched call materialized:
  - active game batch: `round-000043`;
  - checkpoint refs: `4512`;
  - pair count: `300`;
  - game count: `6300`;
  - queue length: `0`;
  - active batch covers current intake: true;
  - active batch is newer than latest rating: true.
- At the check it was only about `29s` old, so no output yet is acceptable.
- Next proof: after a wait, `round-000043` must still be active or have game
  output; it must not be skipped just because more checkpoints arrive.

## 20:56 EDT Update

- `round-000043` survived the quiet interval and started writing output:
  - still active;
  - `4512` checkpoint refs;
  - `300` pairs / `6300` games;
  - liveness probe saw game output;
  - latest sampled output was not stale;
  - queue length remains `0`.
- Modal app list:
  - trainer app still deployed with active tasks;
  - current tournament app has active game tasks;
  - old tournament app instances are stopped or nearly drained.
- This proves the main live path is back through:
  `checkpoint discovery -> intake -> bounded tournament batch -> game output`.
- Remaining proof still needed later:
  `ratings written -> trainer export refresh -> trainers consume that newer export`.

## 20:18 EDT Previous Truth

- The full `cz26-full-20260517a` trainer batch is running again.
- Latest fast trainer status:
  - `136 / 136` heartbeats and progress files;
  - `136` running;
  - `0` failed;
  - checkpoint artifact count range `9..66`;
  - total trainer-visible checkpoint artifacts `5217`.
- The tournament app is alive:
  - latest published rating advanced to `round-000040`;
  - latest rating still covers `4192` tournament checkpoint refs;
  - active game batch is `round-000041`;
  - `round-000041` covers `4192` checkpoint refs, `300` pairs, and `6300`
    games;
  - game output is landing for `round-000041`.
- Important correction: trainer fast-status `checkpoint_count_sum` is not
  automatically the same thing as tournament-visible unique checkpoint refs.
  The run-status scanner also counts mirrored checkpoint artifacts, while
  tournament intake scans only
  `attempts/<attempt>/train/lightzero_exp*/ckpt/iteration_*.pth.tar`.
- The active intake manifest is a live watch over the saved `136` run IDs with
  `checkpoint_selection=all`, plus explicit previously-seen checkpoint refs.
  In principle it should discover new unique checkpoint paths automatically.
- Current hard question:
  - if tournament discovery also sees only `4192` refs, the apparent `5217`
    trainer artifact count is likely duplicate/mirror/resume-path accounting;
  - if tournament discovery sees more than `4192` refs, the scheduled intake
    tick or manifest update path is missing them and must be fixed.
- Trainer consumption of latest export is still **not proven**:
  - `trainer-proof` saw `3792` assignment refresh events;
  - latest-applied target assignment count for export generation `30` is still
    `0`;
  - all `136` latest refresh decisions are currently `unchanged`.
- Plain English: the full trainer batch is running, and the tournament is
  actively rating, but the closed-loop proof is incomplete until we prove
  whether post-relaunch checkpoints are true new tournament refs and until
  trainers visibly load the latest exported opponent assignments.

## 20:00 EDT Previous Truth

- The full `cz26-full-20260517a` trainer batch has been relaunched.
- Relaunch path:
  1. redeployed trainer app
     `curvyzero-lightzero-curvytron-visual-survival-train-v2`;
  2. re-submitted saved manifest
     `artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.json`;
  3. wrote relaunch record
     `artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.relaunch_20260517_1955.json`.
- Relaunch submission result:
  - `136 / 136` rows spawned;
  - `136` train function call ids recorded;
  - `136` poller function call ids recorded;
  - `24` assignment artifacts written;
  - `24` refresh pointers written;
  - training candidate refresh config written for
    `cz26-live-20260517a-elo-cz26-live-20260517a-training`.
- Modal app list now shows:
  - trainer app deployed with `138` tasks;
  - tournament app deployed with `221` tasks.
- Fast trainer status after relaunch:
  - `136 / 136` heartbeats;
  - `136 / 136` progress files;
  - `111` running;
  - `22` completed;
  - `3` failed;
  - checkpoint count range `8..66`;
  - total checkpoint artifacts `5066`.
- Tournament/export state:
  - latest rating remains `round-000038` with `4192` checkpoints;
  - trainer-facing export advanced to generation `30` from `round-000038`;
  - active same-pool rerate remains `round-000040`, `300` pairs / `6300`
    games, with `828` root-completed games and stale sampled output.
- Trainer consumption of generation `30` is **not proven yet**:
  - `trainer-proof` saw `136` runs and `3660` assignment refresh events;
  - latest-applied target assignment count for gen `30` was `0`;
  - this may simply be refresh cadence, but do not call the feedback loop closed
    until this count becomes nonzero and provider-false remains zero.
- Plain English: the full batch is running again, but the closed-loop proof is
  now waiting on trainers to visibly load the latest tournament export and on
  new checkpoints to flow into tournament intake.

## 19:53 EDT Previous Truth

- The full `cz26-full-20260517a` trainer batch is **not currently running**.
  The deployed Modal app list does not include
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`.
- It was launched earlier and produced checkpoints, but the trainer app was
  deliberately stopped around 14:00 EDT to free Modal capacity for tournament
  publish/reduce recovery.
- The tournament/export side recovered after that:
  - latest published rating is `round-000038`;
  - latest rating checkpoint count is `4192`;
  - trainer-facing export is generation `29`;
  - export source is `round-000038`;
  - export has `100` active rows and `24` rewritten assignment pointers.
- Current intake count is `4192`; latest rating count is also `4192`; therefore
  `new_checkpoints_not_in_latest_rating=0` and `queue_len=0` for the current
  checkpoint pool.
- The tournament app is still running a same-pool bounded game batch:
  - active internal batch: `round-000040`;
  - checkpoint count: `4192`;
  - pair count: `300`;
  - game count: `6300`;
  - current root completed count: `828 / 6300`;
  - latest sampled output is stale, so the next tournament action is recovery
    scan/drain handling, not duplicate uncontrolled spawning.
- The full feedback loop is **not currently closed** because no live trainers
  are producing new checkpoints or proving that they consume export generation
  `29`.
- Plain English: the tournament is caught up to the checkpoints it already has,
  but the main trainer batch is stopped. Relaunch is required before we can
  claim the automated trainer -> checkpoint -> tournament -> export -> trainer
  loop is alive again.

## 15:19 EDT Previous Truth

- `round-000038` is still active and still not stale.
- It still covers `4192` checkpoints, `300` pairs, `6300` games.
- Latest sampled output age is about `160s`, so game output is still fresh.
- Sampled game children are still `SUCCESS`; rating loop/round remain pending.
- Latest rating remains `round-000036`; trainer export remains generation `20`
  from `round-000036`.
- Correct action remains wait/recheck. Do not spawn another drain while
  `round-000038` is current and producing output.

## 15:08 EDT Previous Truth

- `round-000038` remains the active batch and still looks useful.
- It covers `4192` checkpoints, `300` pairs, `6300` games.
- Liveness probe saw `21` completed summaries in the sampled scan; latest
  sampled output was about `143s` old.
- The sampled Modal call graph shows:
  - drain call succeeded;
  - rating loop / rating round still pending;
  - sampled tournament game children are `SUCCESS`.
- Latest published rating remains `round-000036`; trainer export remains
  generation `20` from `round-000036`.
- Do not spawn another drain. The correct action is wait/recheck until
  `round-000038` reduces/publishes or becomes stale/partial-reduce due.

## 15:00 EDT Previous Truth

- Immediate state: `round-000038` is now the active bounded tournament batch.
- `round-000038` covers the current intake pool:
  - checkpoint count: `4192`;
  - pairs: `300`;
  - games: `6300`;
  - liveness probe saw real game output within about one minute.
- Latest published rating remains `round-000036` / `3427` checkpoints until
  `round-000038` finishes or partially reduces.
- Trainer-facing export remains current with the latest rating:
  - generation: `20`;
  - source: `round-000036`;
  - active rows: `100`;
  - rewritten assignment pointers: `24`.
- What happened since 14:52:
  - first fresh drain `fc-01KRVMGPC5EZ8THYHQXPDV0CP4` spawned a rating loop that
    never materialized useful active state;
  - app list showed the tournament app had `1583` tasks while status had no
    active batch, so the live compute state was stale/noisy;
  - stopped that app, redeployed current code, and spawned fresh drain
    `fc-01KRVMRCZ6JYSG28GYTFTR5C18`;
  - that drain produced active `round-000038` with real output.
- Current gate:
  1. watch `round-000038` until games complete or partial reduce is due;
  2. prove latest rating advances beyond `round-000036`;
  3. refresh trainer-facing export from that new rating;
  4. relaunch/prove trainers load the refreshed export.

## 14:52 EDT Previous Truth

- Honest state: improved, but not stable enough to relaunch trainers yet.
- Immediate publish/export repair succeeded:
  - `round-000036` published successfully;
  - latest rating is now `round-000036`;
  - latest rating checkpoint count is `3427`;
  - trainer-facing export generation `20` was refreshed from `round-000036`;
  - the export rewrote `24` assignment pointers and currently exposes `100`
    active rows.
- Current intake:
  - `4192` checkpoints have reached tournament intake;
  - `765` checkpoints are newer than the latest published rating and need the
    next bounded tournament batch.
- Current tournament scheduling:
  - config is bounded: `pair_selection=adaptive_v0`, `pairs_per_round=300`;
  - `round-000037` was created for the `4192` checkpoint pool but produced
    zero output and was skipped as an orphan;
  - a fresh bounded drain was spawned at 14:52 EDT:
    `fc-01KRVMGPC5EZ8THYHQXPDV0CP4`.
- Current blockers:
  - trainer consumption from generation `20` is not yet proven because the
    training app was stopped to free capacity;
  - the fresh drain must be checked for a real active batch and real game
    output;
  - we still need a permanent fix so old/bad batches and missing worker
    ownership cannot silently stall the pipeline again.
- Plain English: the rating/export path moved forward. The next checkpoint pool
  still needs to be rated, and trainers must be relaunched only after the live
  tournament lane proves it can keep moving without babysitting.

## 13:55 EDT Previous Truth

- Honest state: not stable. Trainers are still producing checkpoints and
  status/control reads can run, but the tournament publish path is stuck.
- Current active game batch:
  - id: `round-000036`;
  - checkpoint count: `3427`;
  - pairs: `300`;
  - games: `6300`;
  - observed progress: `6289 / 6300` games complete, `299 / 300` pairs
    complete;
  - latest rating still has not advanced from `round-000034`.
- Intake:
  - about `4107` checkpoints;
  - about `1368` newer than latest published rating;
  - about `680` arrived after `round-000036` started, so they belong to the
    next batch.
- Immediate live repair attempts:
  - added a separate reducer `curvytron_rating_reduce_rescue`;
  - lowered reducer resources to `0.25 CPU / 1GB`;
  - added `curvytron_feedback_loop_reduce_rescue` on the same volume set as the
    status/control function;
  - set tournament game worker warm buffers to zero;
  - stopped stale tournament apps and redeployed.
- Current blocker:
  - rescue call `fc-01KRVH57NGQY1VZDSVC8N65886` is still `PENDING` with no task
    id;
  - previous reducer/rescue calls behaved the same;
  - status calls still run, so this is not a total Modal outage.
- 14:00 EDT capacity action:
  - stopped training app `ap-FTBsuB0JXLZoA5MYhadNYv` because it was holding
    `100+` tasks and the tournament subscriber/reducer could not reliably get
    CPU;
  - checkpoints already committed to the v2 volumes are preserved;
  - running trainers are killed and must be relaunched after the tournament
    publish path is proven.
- 14:01 EDT live repair:
  - stopped and redeployed the tournament app after stopping trainers;
  - current tournament app id is `ap-RTOCoNbF6UIYfSNotvgvBS`;
  - invoked embedded `mode=reduce_rescue` through
    `curvytron_feedback_loop_status`; it started but hit the old 300-second
    status timeout.
- 14:08 EDT live repair:
  - raised `curvytron_feedback_loop_status` timeout to 30 minutes;
  - redeployed;
  - re-ran embedded `mode=reduce_rescue`;
  - current local process is waiting for output. Do not interrupt unless a
    stronger signal says it is wedged.
- Current deployed tournament app:
  - app id: `ap-RTOCoNbF6UIYfSNotvgvBS`;
  - tasks were low after zeroing warm buffers, so idle game prewarm is no
    longer the whole explanation.
- Next gate:
  1. get one reducer lane to actually start and publish `round-000036`;
  2. confirm latest rating becomes `round-000036`;
  3. run `refresh-if-ready`;
  4. run `trainer-proof`;
  5. drain the backlog into the next bounded batch.

Do not call this stable until that gate passes.

## 13:12 EDT Previous Truth

- Honest state: not stable yet. The correct active tournament batch exists, but
  the latest published rating has not advanced from it.
- Current active game batch:
  - id: `round-000036`;
  - checkpoint count: `3427`;
  - pairs: `300`;
  - games: `6300`;
  - status: alive with real game output;
  - problem: normal publish waits for every game, and some games are taking
    more than an hour.
- Latest published rating:
  - id: `round-000034`;
  - checkpoint count: `2739`.
- Intake:
  - about `3956` checkpoints;
  - about `1217` checkpoints newer than the latest published rating;
  - about `529` checkpoints arrived after `round-000036` started, so they are
    backlog for the next batch.
- Immediate repair:
  - added bounded recovery scan behavior;
  - added old-active-batch partial-reduce eligibility after one hour of real
    output;
  - deployed the patch;
  - spawned direct detached partial reduce
    `fc-01KRVEMQYPK41S703PJ7PB5DKA`, but it stayed pending in the detached
    local-entrypoint app;
  - stopped the tournament deployed app and the detached local-entrypoint app
    to kill the live game swarm and free the control/reduce lane;
  - redeployed the patched tournament app;
  - spawned a fresh deployed-app partial reduce:
    `fc-01KRVF11V4F7NVABHYDAWYY8S9`.
- Follow-up fix now in code:
  - game worker warm containers were reduced from `min=100, buffer=400` and
    `min=500, buffer=500` to `min=0, buffer=16` for both game paths;
  - reason: fanout is still possible, but redeploy should not reserve hundreds
    of idle game workers before reduce/control work can start.
  - reduce resources were reduced from `2 CPU / 8GB` to `1 CPU / 2GB`.
  - latest active reduce call after these patches:
    `fc-01KRVFHF8025JT906B4M7VGXXP`.
- Current gate:
  1. wait for `fc-01KRVF11V4F7NVABHYDAWYY8S9` to write ratings for
     `round-000036`;
  2. confirm latest rating becomes `round-000036`;
  3. run `refresh-if-ready`;
  4. run `trainer-proof`;
  5. drain the backlog into the next bounded batch.

## 11:46 EDT Stop/Redeploy Repair

- Honest state: the loop is not stable yet, but the immediate old-pool blocker
  has been moved out of the way.
- `round-000034` is the latest published rating:
  - checkpoint count: `2739`;
  - game count: `28308 / 28308`;
  - trainer-facing export: generation `17`, sourced from `round-000034`.
- Intake has `3427` checkpoints, so `688` checkpoints are still newer than the
  latest published rating.
- `round-000035` was the wrong next game batch:
  - it was bounded correctly at `300` pairs / `6300` games;
  - but it only had `2739` checkpoints, exactly the already-rated
    `round-000034` pool;
  - it did not include the `688` newer refs.
- The control patch skipped `round-000035`, but Modal logs still showed old
  `round-000035` game workers running and hitting `600s` function timeouts.
  Plain English: state said "skip it", but compute from that bad batch was still
  alive.
- Immediate repair performed:
  - stopped deployed tournament app `ap-NOZz2IQRfFlBUbz0zIRC2M`;
  - redeployed `curvyzero-checkpoint-tournament-v2`;
  - status after redeploy showed no active game batch and the stale child rating
    call terminated;
  - spawned one fresh detached drain:
    `fc-01KRV9TXJHWSVXF2ZY046RQP4F`.
- Current gate:
  1. done: the fresh drain created `round-000036`;
  2. done: `round-000036` covers `3427` checkpoints, not `2739`;
  3. done: it is bounded at `300` pairs / `6300` games;
  4. game output is landing; pending rating publication;
  5. current latest export proof done for `round-000034`; pending repeat after
     `round-000036` publishes.
- Trainer proof after generation `17`:
  - target assignment count: `24`;
  - provider-ok rows: `39218`;
  - provider-false rows: `0`;
  - target assignment latest-applied count: `48 / 136`.
- 12:01 EDT status:
  - `round-000036` is still active;
  - active batch still has `3427` checkpoints, `300` pairs, `6300` games;
  - liveness sample found game output, but the slow full progress probe timed
    out at `300s`;
  - compact call graph reports the visible `curvytron_tournament_game` children
    as `SUCCESS`, while `curvytron_rating_round` is still `PENDING`;
  - intake advanced to `3622`, so `195` checkpoints arrived after
    `round-000036` started. That is backlog for the next batch, not proof this
    batch is wrong.
- 12:08 EDT status:
  - compact status still showed `round-000036` active on the right `3427`
    checkpoint pool;
  - its sampled output looked stale, so the correct action was a recovery scan,
    not a duplicate normal drain;
  - spawned detached recovery scan `fc-01KRVB3CVZH2VHH7R0EJS0ZST5`;
  - intake is now `3652`, so `225` checkpoints are backlog for the next batch.
- 12:17 EDT status:
  - `round-000036` is still active and still on the correct `3427` pool;
  - liveness sample is fresh again, so do not skip it;
  - recovery scan `fc-01KRVB3CVZH2VHH7R0EJS0ZST5` is still pending;
  - latest rating remains `round-000034`;
  - trainer export generation advanced to `18`, but it is still sourced from
    `round-000034`, so it is not new tournament progress.
- 12:28 EDT status:
  - recovery scan `fc-01KRVB3CVZH2VHH7R0EJS0ZST5` timed out at `600s`;
  - `round-000036` itself is still alive with fresh sampled game output;
  - latest rating is still `round-000034`;
  - intake has `3727` checkpoints, so `300` checkpoints are now backlog after
    `round-000036` started;
  - permanent tooling issue: full recovery/progress scans are too expensive and
    need an incremental progress index or bounded summary artifact.
- 12:39 EDT status:
  - `round-000036` still has the correct `3427` checkpoint pool and fresh
    sampled game output;
  - latest rating is still `round-000034`;
  - intake has `3789` checkpoints, so `362` checkpoints are backlog for the
    next batch;
  - current conclusion: slow but not dead.

Do not describe the system as stable until this gate passes.

## 10:30 EDT Reorientation

- Honest state: the full loop is running, but it is not caught up and not fully
  proven for the newest checkpoints.
- Main batch:
  - launch manifest: `cz26-full-20260517a`;
  - rows: `136` (`96` Grid A, `40` Grid B);
  - submit records: `136/136` spawned;
  - fast trainer status: `136/136` heartbeats and progress files;
  - running: `134`;
  - failed: `2` (`cz26a-r013-out67-n0-imm0-b20w05r1`,
    `cz26b-r028-out50-n10-imm10-b20w05r1`);
  - trainer checkpoint artifacts seen by fast status: `3185`;
  - latest progress iteration range: `30000..250000`.
- Tournament intake:
  - current intake checkpoint count: `3193`;
  - latest published rating still covers `2192` checkpoints from
    `round-000033`;
  - checkpoints newer than latest rating: `1001`;
  - queue length: `388`;
  - live config is bounded: `pair_selection=adaptive_v0`,
    `pairs_per_round=300`.
- Current active tournament game-batch artifact:
  - id: `round-000034`;
  - created from `2739` checkpoints;
  - pairs: `1348`;
  - games: `28308`;
  - status: still running, not published;
  - latest cheap liveness sample saw output about `44s` old at 10:30 EDT;
  - app logs show successful `round-000034` game results at 10:29 EDT.
- Trainer-facing export and consumption:
  - current trainer export is generation `13`;
  - it still sources `round-000033`, not `round-000034`;
  - `trainer-proof` saw `102/136` runs with a current generation-13 target
    assignment as latest-applied;
  - provider rows for generation-13 targets: `74170 ok`, `0 false`,
    `153090 null/pending`.
- Tooling gap:
  - the expensive `--progress-probe` timed out at `300s`;
  - cheap liveness and logs prove games are landing, but we still do not have a
    cheap total count like "N of 28308 games complete" for the active batch.
- Plain English timeline:
  - the system was not healthy for the whole past 12 hours;
  - one complete pass was proven at 08:52 EDT when `round-000033` published;
  - since 09:08 EDT, `round-000034` has been the next active tournament
    game-batch;
  - trainers have continued producing checkpoints while `round-000034` runs;
  - the current bottleneck is tournament game execution/reduction, not trainer
    launch and not trainer consumption.
- Next proof needed:
  1. `round-000034` publishes a rating beyond `2192`.
  2. `refresh-if-ready` writes a trainer export sourced from `round-000034`.
  3. `trainer-proof` sees running trainers load that newer export.
  4. A later status shows the queued checkpoints get picked up by the next
     bounded game-batch.

## 11:20 EDT Recovery Fix And Current State

- The system was not "relatively stable" before this pass. The trainer side was
  connected, but the tournament side had two real design/control problems:
  1. stale active game-batches were blocked by the control plane instead of
     being allowed to enter recovery;
  2. `pairs_per_round=300` was not a hard cap because adaptive scheduling could
     expand the round to half of the undercovered checkpoint pool.
- Code fixes made and deployed:
  - stale active batches can now trigger a recovery scan instead of being
    blocked as merely "active";
  - the scheduled drain tick now uses a small activity probe so it can notice a
    stale active batch;
  - future adaptive batches respect the requested `pairs_per_round` hard cap;
  - large-batch recovery uses a bounded activity probe instead of a full
    28k-game summary scan;
  - operator output now includes the recovery skip decision.
- Focused tests:
  - `19 passed` for adaptive/control/recovery slices;
  - `18 passed` after scheduled tick and summary visibility changes;
  - changed Python files compile.
- Live recovery result for `round-000034`:
  - recovery did **not** skip or reduce it;
  - reason: `not_skippable`;
  - scan mode: `bounded_activity_probe`;
  - completed games found: `5103 / 28308`;
  - started pairs found: `243 / 1348`;
  - latest real activity was fresh at scan time, so the batch is slow, not dead.
- Visible status after recovery:
  - root progress now reports `4414 / 28308` games complete, about `15.6%`;
  - `round-000034` is still active and unpublished;
  - latest published rating is still `round-000033` / `2192`;
  - intake is now `3401`, so `1209` checkpoints are newer than latest rating.
- Plain English: do not abandon the current batch yet; it is still producing
  real outputs. The deeper fix is that future batches should not be this large,
  and stale active batches now have an explicit recovery path.

## 11:25 EDT Status And Handoff Update

- A plain-language handoff now exists:
  `WHAT_THE_HELL_IS_GOING_ON_HANDOFF_2026-05-17.md`.
- Latest compact status at `15:23:19Z`:
  - intake checkpoint count: `3427`;
  - latest published rating: `round-000033` / `2192`;
  - checkpoints newer than latest rating: `1235`;
  - queue length: `622`;
  - active game-batch artifact: `round-000034`;
  - active batch still covers `2739` checkpoints, `1348` pairs, and `28308`
    games;
  - root progress still shows `4414 / 28308` completed games;
  - trainer-facing export is generation `15`, still sourced from
    `round-000033`.
- The cheap liveness sample is stale again, so the correct action is the
  patched recovery scan. A `drain-if-ready` recovery scan is currently running.
- Plain English: the system is more protected than before, but not "stable" yet.
  Current proof is still stuck at `round-000033` until `round-000034` either
  publishes or is safely skipped and replaced by a bounded batch.

## 11:26 EDT Round 34 Published

- The patched recovery/drain command found that `round-000034` had completed
  and published:
  - latest published rating: `round-000034`;
  - latest rating checkpoint count: `2739`;
  - completed games: `28308 / 28308`;
  - pairs: `1348`;
  - checkpoints newer than latest rating: `688`.
- The same command repaired/claimed the next desired pool and spawned rating
  call `fc-01KRV8JHYV02SJ8GNH1SJPGQW4` for `3427` checkpoints.
- Trainer-facing export was still generation `15` from `round-000033` at that
  moment, so `refresh-if-ready` and then `trainer-proof` are now the proof
  steps.
- Plain English: tournament publication moved forward. The remaining immediate
  proof is that trainers consume the new `round-000034` export, and that the
  next spawned tournament batch is bounded by the hard-cap fix.

## 11:31 EDT Old-Pool Batch Fix

- `refresh-if-ready` succeeded:
  - trainer-facing generation: `16`;
  - source rating: `round-000034`;
  - active rows: `100`;
  - rewritten pointers: `24`.
- Status then showed the next active game-batch `round-000035`:
  - pair count: `300`;
  - game count: `6300`;
  - so the hard pair cap worked.
- But `round-000035` had the wrong pool:
  - active batch checkpoint count: `2739`;
  - latest rating checkpoint count: `2739`;
  - intake checkpoint count: `3427`;
  - missing newer checkpoint refs: `688`.
- Local artifact check confirmed:
  - `/private/tmp/cz26_round35_input.json` has `2739` input refs;
  - `/private/tmp/cz26_intake_manifest.json` has `3427` refs;
  - the `688` missing refs are in intake but absent from `round-000035`;
  - `round-000035` input refs equal the latest `round-000034` rating refs.
- Code patch made this non-blocking:
  - control decisions now allow recovery scan for
    `active_game_batch_not_covering_new_checkpoints`;
  - skip decision can skip an active round with reason
    `different_spec_already_rated_pool` when the desired pool includes newer
    checkpoints but the active input is no newer than latest published rating.
- Focused tests passed:
  - feedback-loop control decision slice: `4 passed`;
  - old-pool/smaller-pool skip decision slice: `3 passed`;
  - changed Python files compile.
- Deploy is in progress. Next: run `drain-if-ready`, verify `round-000035` is
  skipped/dropped or moved past, then verify the next batch covers the new refs.

## 09:08 EDT Reorientation

- The whole loop is not "fully stable for 12 hours." The honest state is:
  one complete loop has been proven, and the next loop is currently running.
- Proven loop:
  - trainer checkpoints reached intake;
  - tournament game-batch artifact `round-000033` rated `2192` checkpoints;
  - it completed `22575` games and published as latest rating;
  - `refresh-if-ready` wrote trainer-facing generation `11` from
    `round-000033`;
  - `trainer-proof` found generation-11 assignment hashes in real trainer
    provider rows, with `3/136` runs already showing a target assignment as
    latest-applied at the time of proof.
- Current in-flight loop:
  - after app cleanup/redeploy, one fresh drain request
    `fc-01KRV0JK89X9FFFYHHCZEMFNBJ` spawned rating loop
    `fc-01KRV0JQYM98FDTZD6Q55VY04B`;
  - it created internal game-batch artifact `round-000034`;
  - `round-000034` covers `2739` checkpoint rows, `1348` pairs, and `28308`
    games;
  - liveness probe saw fresh game output, so games are landing;
  - latest rating is still `round-000033` / `2192` until `round-000034`
    reduces and writes ratings.
- Current intake has advanced to `2805`, so `66` checkpoints arrived after
  `round-000034` was created. That is normal backlog for the next batch, not
  proof that the current batch is wrong.
- Plain English timeline:
  - before 08:52 EDT, the live system was stuck behind stale/bad live state and
    slow/retried game batches;
  - at 08:52 EDT, `round-000033` finally published and proved one full feedback
    pass through trainer export and partial trainer consumption;
  - at about 09:03 EDT, a new drain was started after stopping/redeploying the
    crowded app state;
  - by 09:08 EDT, `round-000034` was active and producing game output.
- Do not spawn duplicate drains while `round-000034` is active. Next proof is:
  `round-000034` publishes beyond `2192`, then trainer export refreshes from it,
  then `trainer-proof` sees trainers load that newer generation.

## 09:25 EDT Trainer And Tournament Check

- Main launch artifact is real:
  - manifest: `cz26-full-20260517a`;
  - rows: `136`;
  - Grid A: `96`;
  - Grid B: `40`;
  - Grid B includes pure slot controls `b100`, `w100`, and `r1`;
  - submit records: `136/136` status `spawned`;
  - all rows use the same pinned old r18fresh rank-1 initial checkpoint.
- Fast trainer status tooling was added because the old all-run status table
  timed out after `300s` by scanning too much per run.
- Fast trainer check now says:
  - rows checked: `136`;
  - heartbeats: `136`;
  - progress files: `136`;
  - running: `134`;
  - failed: `2`;
  - failed run ids:
    `cz26a-r013-out67-n0-imm0-b20w05r1`,
    `cz26b-r028-out50-n10-imm10-b20w05r1`;
  - failed-row detail:
    - `cz26a-r013...` failed after checkpoint `iteration_170000` with `36`
      checkpoints;
    - `cz26b-r028...` failed after checkpoint `iteration_30000` with `8`
      checkpoints;
    - current heartbeat/status files do not expose a failure reason string;
  - checkpoint count range per run: `8..36`;
  - total checkpoint artifacts seen by trainer status: `2910`;
  - latest progress iteration range: `30000..220000`.
- Tournament status at `13:25:57Z`:
  - latest rating still `round-000033` / `2192`;
  - active game-batch artifact still `round-000034`;
  - `round-000034` covers `2739` checkpoints, `1348` pairs, and `28308`
    games;
  - liveness probe saw `21` completed game summaries, with newest output about
    `157s` old;
  - current intake is `2877`, so `138` checkpoints arrived after
    `round-000034` was created;
  - queue length is `72`.
- Plain English: trainers are mostly running and producing many checkpoints;
  the tournament is actively rating a large slice of them, but latest published
  rating has not advanced beyond `2192` yet. The next automatic feedback pass is
  still in progress, not complete.
- Separate `cz26c` canary/control note:
  - run id: `cz26c-r001-out100-n0-imm0-b20w05r1`;
  - status: `completed`;
  - checkpoint count: `350`;
  - latest progress iteration: `17385`.

## 09:47 EDT Follow-Up

- After a 15-minute wait, `round-000034` is still active and not published.
- Tournament status:
  - latest rating remains `round-000033` / `2192`;
  - active game-batch artifact remains `round-000034`;
  - active batch still covers `2739` checkpoints, `1348` pairs, and `28308`
    games;
  - current intake is `2989`;
  - `250` checkpoints arrived after `round-000034` was created;
  - queue length is `184`;
  - liveness probe with activity enabled saw fresh output about `128s` old.
- Trainer-facing export advanced to generation `12`, still sourced from
  `round-000033`. This is not a new tournament result, but it is the current
  trainer assignment generation.
- Trainer-proof for generation `12`:
  - runs scanned: `136`;
  - target assignment SHAs: `24`;
  - latest-applied target count: `62`;
  - assignment refresh applied count: `1413`;
  - target env rows: `138045`;
  - provider OK rows: `48666`;
  - provider false rows: `0`;
  - provider null/pending rows: `89379`.
- Plain English: the trainer side is definitely loading current tournament
  export assignments. The missing proof is still the next tournament publish:
  `round-000034` must reduce and write a rating beyond `2192`.

## 10:19 EDT Follow-Up

- After a 30-minute wait, `round-000034` is still active and not published.
- Tournament status:
  - latest rating remains `round-000033` / `2192`;
  - active game-batch artifact remains `round-000034`;
  - active batch covers `2739` checkpoints, `1348` pairs, and `28308` games;
  - active batch age is about `3604s` / `60m`;
  - current intake is `3136`;
  - `397` checkpoints arrived after `round-000034` was created;
  - queue length is `331`;
  - liveness probe saw output about `341s` old, under the `600s` stale line.
- Trainer-facing export advanced to generation `13`, still sourced from
  `round-000033`; this is assignment refresh churn from the same latest rating,
  not a new tournament publish.
- Plain English: the live system is still moving, but the tournament game batch
  is the slow leg. Do not start another drain while `round-000034` is active.

## 08:52 EDT Loop Proof Update

- `round-000033` finished and published:
  - latest rating is now `round-000033`;
  - rated checkpoint count: `2192`;
  - completed games: `22575`;
  - pairs: `1075`;
  - published around `12:49:31Z`.
- Trainer-facing refresh succeeded:
  - generation: `11`;
  - snapshot: `auto-r000033-g11-50405322`;
  - source game-batch artifact: `round-000033`;
  - source row count: `2192`;
  - active trainer-facing rows: `100`;
  - rewritten pointers: `24`.
- Trainer consumption proof found generation-11 assignments in real trainer
  logs:
  - runs scanned: `136`;
  - target assignment SHAs: `24`;
  - latest-applied target count: `3`;
  - target env rows: `6663`;
  - provider OK rows: `4548`;
  - provider false rows: `0`;
  - provider null/pending rows: `2115`.
- Plain English: we have now seen the core feedback path happen at least once:
  tournament published a newer rating, the trainer-facing export was refreshed
  from it, and some running trainers loaded generation-11 opponents. It is not
  fully caught up across all trainers yet.
- Current live backlog after publish:
  - intake checkpoint count: `2714`;
  - latest rated checkpoint count: `2192`;
  - queued/newer checkpoints: `522`.
- Next bounded drain was spawned for those queued checkpoints:
  - function call id: `fc-01KRV02A38VYDJPE3JWM4A0W4M`;
  - requested at `2026-05-17T12:55:05Z`;
  - new checkpoints at request: `522`;
  - queue length at request: `522`.
  The drain returned and spawned rating loop
  `fc-01KRV02Z7H8Q7466J50EPSPZ7E` for the `2714`-checkpoint desired pool.
- Immediate follow-up status:
  - queue length is now `0`;
  - no active game-batch artifact is visible yet;
  - the rating loop call is still pending.
- Logs say the rating loop is waiting to be scheduled on a CPU worker. Plain
  English: Modal capacity/scheduling is the current wait, not missing intake
  and not a reason to spawn another drain.
- Next proof is that this pending rating loop creates a new active game-batch
  artifact and eventually publishes beyond `2192`.
- Local control-script guard tightened: before spawning a new drain after the
  operator lease ages out, the script now probes the previous function call and
  blocks if that call or any child in its compact call graph is still pending.
  This prevents duplicate drains while `fc-01KRV02Z7H8Q7466J50EPSPZ7E` is
  waiting for capacity.
- Tooling note: the first full progress scan was too heavy and timed out after
  `300s`. The deployed/local default is now fast status; the expensive progress
  scan is explicit `--progress-probe`.

## 08:39 EDT Reorientation

- Live status still says the tournament is active, not finished:
  - active internal game-batch artifact: `round-000033`;
  - checkpoint inputs/spec/roster rows: `2192`;
  - pairs: `1075`;
  - intended games: `22575`;
  - intake checkpoint count: `2668`;
  - queue length: `476`;
  - checkpoints newer than latest rating: `1749`.
- Latest published rating is still `round-000015` / `919`.
- Trainer-facing export is generation `10`, but it still sources
  `round-000015`. This is an old-rating export refresh, not proof of the live
  tournament feeding trainers with new winners.
- Important status-tool caveat:
  - `probe_completed_game_count=21` is only a narrow liveness sample. It is not
    the total number of completed games.
  - `completed_game_count=0` on the root game-batch summary means the batch has
    not reduced/written root progress yet. It does not mean zero per-game
    outputs exist.
  - the Modal call-graph summary is capped/sample-like (`node_count=100`), so
    `96` sampled game successes is not proof that all `22575` games completed.
- App logs at `12:36-12:39Z` show fresh `round-000033` game completions plus
  repeated runner-disappeared/reschedule messages. Plain English: game workers
  are still completing and being retried.
- Current conclusion: the loop is still not validated. The current blocker is
  tournament game throughput/retries and lack of a clear total-progress
  readout, not checkpoint intake and not old tournaments.
- Do not spawn another drain while `round-000033` is active. Next work:
  improve the status/progress tooling so operators can see true sampled/total
  progress, then keep monitoring. If a rating beyond `919` publishes, run
  `refresh-if-ready` and then `trainer-proof` immediately.
- Local tooling patch is ready and focused tests pass:
  - `curvytron_feedback_loop_status` can now include a `progress_probe`;
  - `scripts/curvytron_live_loop_control.py` separates
    `root_completed_game_count`, `liveness_probe_*`, and `progress_probe_*`;
  - focused test:
    `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "feedback_loop_status or rating_game_batch_status_summary"` -> `7 passed`.
  This is not deployed until we intentionally update the live app.

## 08:31 EDT Log Check

- App logs confirm `round-000033` games were still finishing at
  `12:30-12:31Z`.
- The active batch has not reduced because game execution is still in progress,
  not because intake is missing.
- Logs include Modal runner disappearance / reschedule messages. Plain English:
  some in-progress game workers are being retried.
- Logged game results have normal-looking game outcomes and balanced-random
  seats, but worker wall times are very large for low physical-step counts.
  This is now a throughput/retry concern to keep watching.
- Current conclusion: do not spawn another drain. Wait for the remaining game
  workers or recovery path. If progress stops, debug tournament game throughput
  and retry behavior.

## 08:26 EDT Update

- Short recheck after user reorientation: `round-000033` is still active and
  latest rating is still `round-000015` / `919`.
- Important progress: the sampled Modal call graph now shows all `96`
  sampled tournament game children under the rating round as `SUCCESS`.
- Remaining pending pieces in the call graph are the rating loop and rating
  round themselves.
- The narrow status probe is fresh again:
  - probe completed summaries `21`;
  - newest probed output age about `112s`.
- Interpretation correction from the 08:39 recheck: this does not prove all
  games are complete because the call graph is capped/sample-like. Keep treating
  the active issue as game execution/retries until root progress or a newer
  rating proves reduction.

## 08:20 EDT Update

- Five-minute follow-up: `round-000033` is still active and not published.
- Narrow output probe is now stale:
  - probe completed summaries still `21`;
  - newest probed output age about `944s`;
  - status flag includes `active_game_batch_output_stale`.
- But the Modal call graph still shows real progress:
  - sampled tournament game successes increased from `67` to `75`;
  - sampled pending tournament game calls decreased from `29` to `21`.
- Interpretation: do not conclude the whole batch is stuck from the narrow
  output probe alone. The live function calls are still finishing games. Keep
  rechecking on a short cadence; if call-graph progress stops too, then debug
  active games/reduce.

## 08:15 EDT Update

- After a 30-minute wait, `round-000033` is still active and has not published.
- Status facts:
  - latest published rating remains `round-000015` / `919`;
  - intake is now `2539`;
  - queue length is `347`;
  - active batch still has `2192` checkpoints, `1075` pairs, and `22575`
    games;
  - compact sampled Modal call graph moved to `67` tournament game successes
    and `29` tournament game calls pending;
  - provisional loop is now `SUCCESS`, rating loop/rating round are still
    pending;
  - newest probed game output age is about `599s`, near the stale threshold.
- Interpretation: progress is real but slow. Do a shorter follow-up recheck
  rather than another long sleep. If output age keeps growing past the recovery
  threshold or recovery skips the batch, investigate that as a progress/reduce
  issue; do not start a duplicate drain while the active batch exists.

## 07:44 EDT Update

- After another 15-minute wait, `round-000033` is still active and still has
  not published a rating beyond `919`.
- Status facts:
  - latest published rating: still `round-000015` / `919`;
  - active batch: still `round-000033`, `2192` checkpoint inputs, `1075`
    pairs, `22575` games;
  - current intake: `2406` checkpoints;
  - current queue length: `214`;
  - newest probed output age: about `19s`, so the batch is not stale;
  - compact sampled Modal call graph remains `45` tournament game successes
    and `51` tournament game calls pending.
- Trainer-facing refresh advanced to generation `8`, but it still sources
  `round-000015`. This is not new tournament progress.
- Plain English: the tournament is still playing long-running games. The full
  loop remains unproven until this or a later current batch publishes beyond
  `919`, then trainer refresh and trainer-proof advance from that newer rating.

## 07:28 EDT Update

- After a 10-minute wait, `round-000033` is still active and has not published
  a new rating.
- Compact status facts:
  - latest published rating is still `round-000015` / `919`;
  - active batch remains `round-000033` with `2192` checkpoint inputs,
    `1075` pairs, and `22575` games;
  - current intake has advanced to `2308`, with queue length `116`;
  - active batch still covers checkpoints newer than the latest rating;
  - broad activity probe still sees `21` completed game summaries;
  - newest probed output age is about `203s`;
  - compact Modal call graph shows `45` sampled tournament game calls succeeded
    and `51` sampled tournament game calls still pending.
- Status has no hard blockers. It reports normal open items:
  new checkpoints arrived after this active batch started, and latest rating is
  still behind intake.
- Interpretation: the tournament is not done; do not duplicate the drain. The
  slow part now appears to be waiting for remaining long-running games/reduce.

## 07:15 EDT Update

- Recheck after a 3-minute wait shows `round-000033` is alive and producing
  game output.
- Status facts:
  - active internal game-batch artifact: `round-000033`;
  - active batch checkpoints: `2192`;
  - current intake checkpoints: `2252`;
  - queued checkpoints: `60`;
  - latest published rating: still `round-000015` / `919`;
  - checkpoints newer than latest rating: `1333`;
  - active batch probe completed game summaries: `21`;
  - newest probed game result age: about `11s`;
  - Modal call summary: `44` tournament game calls succeeded, `52` pending.
- Plain English: the current batch is playing games. The `60` queued
  checkpoints arrived after this batch started; they are not proof of a stall.
- Still unproven: rating publish beyond `919`, trainer-facing refresh from that
  newer rating, and trainer consumption of that newer export.
- Tooling cleanup after this check: status `blockers` now means real blockers;
  normal backlog such as new checkpoints arriving during an active batch is
  reported as `open_items`.
- Next action: wait/recheck. Do not spawn another drain while `round-000033`
  is active.

## 07:10 EDT Update

- Current live status was checked through the operator tool, not by raw Volume
  browsing:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 64`.
- The tournament is active on internal game-batch artifact `round-000033`.
  Plain English: this is one current tournament work batch, not a training
  round.
- `round-000033` contains:
  - `2192` checkpoint inputs;
  - `2192` rating-spec checkpoints;
  - `2192` roster rows;
  - `1075` policy pairs;
  - `22575` games.
- This batch covers the full current intake pool at the time of the check:
  intake checkpoint count `2192`, missing-from-intake count `0`.
- Latest published rating is still `round-000015` / `919`, so `1273`
  checkpoints are newer than the latest published tournament rating.
- Trainer-facing refresh generation is still `7` from `round-000015`. This is
  not new progress.
- At age about `57s`, `round-000033` had no visible completed game output yet.
  The function-call graph showed the rating/game calls pending, so the right
  action is to wait and recheck, not start a duplicate drain.
- Next command should be compact status, preferably without the huge call graph
  unless debugging scheduler state:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 64 --no-drain-call-probe`.
- Tooling cleanup made after this check: the default drain-call probe in
  `scripts/curvytron_live_loop_control.py` now summarizes the Modal call graph
  with counts and a small sample instead of dumping every pending game call.
  Use `--full-drain-call-graph` only when the full nested graph is really
  needed.
- Same tooling pass added `status_summary.proof_chain` and
  `status_summary.blockers`, so the operator output directly names the five
  boundaries: intake, active tournament game batch, latest rating, trainer
  export, and trainer consumption proof.
- Do not claim the feedback loop is working until a rating beyond `919`
  publishes, trainer-facing refresh uses that rating, and `trainer-proof`
  proves running trainers loaded the refreshed opponents.

## 06:44 EDT Update

- New live truth: the deployed tournament app was stopped to break an overloaded
  stale scheduling state.
- Why: after the current-code drain request, the rating child call stayed
  pending. Modal system logs for the deployed tournament app showed repeated
  worker preemption / runner-disappeared messages, and the app had thousands of
  outstanding tasks. A fresh current-code rating child could not start cleanly
  inside that app state.
- Plain English: the code fix may be right, but the running app state was
  unhealthy. Code being deployed was not the same thing as the live pipeline
  being repaired.
- Do not trust the current child call
  `fc-01KRTR2B7BPV74FCAY1DPMD68M` after the app stop. Treat it as stale until
  a fresh status call proves otherwise.
- Next action is not raw Volume digging. Next action is:
  `uv run --extra modal modal deploy src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
  followed by:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 64`.
- If that status says no active game batch and the loop is ready or queued for
  drain, run exactly one fresh `drain-if-ready` from the redeployed app.
- The loop is still not validated. Latest known published rating is still
  `round-000015` / `919`; trainer generations sourced from that rating are not
  new tournament progress.
- Redeploy completed successfully at 06:44 EDT. The tournament browser URL is
  again:
  `https://modal-labs-shankha-dev--curvyzero-checkpoint-tournament--93d419.modal.run/`.
- Next command is the deployed-function status probe, not manual artifact
  inspection:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 64`.
- Fresh status after redeploy:
  - status `ready_for_next_rating_batch`;
  - no active game batch;
  - queue length `0`;
  - intake checkpoint count `2121`;
  - latest rating still `round-000015` / `919`;
  - `1202` checkpoints are newer than the latest rating;
  - trainer refresh generation `7` still sources `round-000015`, so it is not
    new rating progress.
- The previous drain request is recorded as `drain_returned`, but it did not
  leave an active game batch. Therefore the next action is exactly one fresh
  `drain-if-ready` from the redeployed app.
- Fresh `drain-if-ready` ran from the redeployed app:
  - call id `fc-01KRTRQJBC6TTPYS1DYZ6RVMBF`;
  - rating checkpoint count `2121`;
  - desired rating-spec checkpoint count `2121`;
  - latest rating checkpoint count `919`;
  - desired new checkpoint count `1202`;
  - pool hash `f2c3fe1079f6222f`;
  - stale rating claim repaired `true`.
- Next proof: immediate status must show a new active game batch from this
  current-code call, or explain why it did not start.
- Immediate status still showed no active game batch. Modal call-graph
  inspection showed `fc-01KRTRQJBC6TTPYS1DYZ6RVMBF` was pending, and app logs
  exposed the real code bug: `curvytron_rating_round` crashed in
  `select_adaptive_v0_pair_slots` with `OverflowError: math range error`.
- Root cause: adaptive scheduler top-band scoring used a naive sigmoid. With a
  large live roster, low-ranked checkpoints passed a large negative value into
  `math.exp`, which overflows before any game batch can be written.
- Fix made locally: stable sigmoid implementation plus a regression test with
  a `1600` checkpoint adaptive roster.
- Local validation:
  - targeted overflow proof: `2 passed`;
  - adaptive scheduler focused sweep: `18 passed`.
- Next action: stop the currently deployed crashing app state and redeploy the
  fixed scheduler, then run status/drain again.
- Stopped crashing deployed app `ap-IrdpT1NJRAr9T3O26dkJFt`.
- Redeployed `curvyzero-checkpoint-tournament-v2` with the stable sigmoid fix.
- Next action: run deployed-function `status`. If it still reports no active
  batch and `ready_for_next_rating_batch`, run one fresh `drain-if-ready`.
- First drain after the fix was blocked by the local operator lease from the
  old crashed call. That was expected but not useful because the old app had
  been stopped.
- Used `--ignore-drain-request-lease` exactly once. Fresh fixed-code drain:
  - call id `fc-01KRTRZ6EYAMPTMX79753ZNN8M`;
  - rating checkpoint count `2148`;
  - desired new checkpoint count `1229`;
  - pool hash `fcc777568a1b540c`.
- Follow-up status now shows real progress:
  - status `rating_game_batch_active`;
  - active internal game-batch artifact `round-000032`;
  - checkpoint/spec/roster counts all `2148`;
  - pair count `1053`;
  - game count `22113`;
  - phase `game_map_started`;
  - active batch is newer than latest rating `919`;
  - missing-from-intake count `0`.
- Plain English: the overflow fix unblocked scheduling. The tournament is now
  actually trying to play the current checkpoint pool. This still does not
  prove the loop is complete until this batch writes ratings, the trainer-facing
  export advances, and trainers load the new export.
- After a 2-minute wait, status and logs prove the active batch is really
  running:
  - status remains `rating_game_batch_active`;
  - active batch is still `round-000032`;
  - intake has advanced to `2161`, so `13` newer checkpoints arrived after this
    batch started;
  - probe found `21` completed game summaries with newest output about `35s`
    old;
  - logs show successful games with `ok=true`, balanced randomized seat order,
    and `max_steps=1048576`.
- Latest rating is still `round-000015` / `919`. Current action is wait for
  completion or reduce; do not spawn another drain while `round-000032` is
  active.
- Five-minute recheck found the next bug:
  - `round-000032` was skipped as `different_spec_zero_output`;
  - skip record had `skip_scan_output_progress=true` but
    `skip_completed_game_count=0`;
  - app logs showed real successful game workers before the skip.
- Plain English: games were running, but the recovery function used a stale
  Modal Volume view when deciding whether output existed. It skipped because it
  did not reload the tournament Volume before scanning outputs written by other
  workers.
- Fix made locally: `_rating_round_skip_decision` reloads the tournament Volume
  before output scanning when scanning the real tournament mount.
- Local validation:
  - skip/status focused slice: `5 passed`;
  - adaptive + skip/status broader focused slice: `23 passed`.
- Next action: stop the current bad deployed app state, redeploy the
  Volume-reload fix, then run one fresh drain if status is ready.
- Stopped bad deployed app state `ap-aAZ5lOtiAfWmsf828BSY0Y`.
- Redeployed `curvyzero-checkpoint-tournament-v2` with the Volume-reload
  recovery fix.
- Next command: compact deployed-function status, then one fresh drain if ready.

## 06:35 EDT Update

- After another wait, `round-000031` was also skipped as
  `zero_progress_smaller_pool`.
- Its skip record has `skip_scan_output_progress=false`, same as the older
  skipped batches.
- Plain English: the just-deployed recovery fix did not get a chance to act on
  `round-000031`; that batch was still under an old in-flight rating/drain
  function call.
- Current status is now `ready_for_next_rating_batch`:
  - latest rating still `round-000015` / `919`;
  - intake `2041`, so `1122` checkpoints newer than latest rating;
  - queue length `0`;
  - no active game batch.
- Next action: run exactly one `drain-if-ready` through the deployed control
  script so the next rating batch is created by the current code.
- Ran that drain:
  - rating call id `fc-01KRTR2B7BPV74FCAY1DPMD68M`;
  - rating checkpoint count `2045`;
  - desired rating-spec checkpoint count `2045`;
  - latest rating checkpoint count `919`;
  - desired new checkpoint count `1126`;
  - pool hash `a4ba214df12ac77d`;
  - stale claim repaired `true`.
- Next proof: a new active batch must appear from the current deployed code.
  If it later skips, its skip record should now have
  `skip_scan_output_progress=true`; otherwise the old in-flight-code problem
  is still present somewhere.
- One minute later, no active batch had appeared. The new child rating call was
  still pending in Modal.
- The previous stale child rating call
  `fc-01KRTNPSZY7MCMPFMQJ9N1EAQM` was also still pending, so it was cancelled.
  The newer current-code child `fc-01KRTR2B7BPV74FCAY1DPMD68M` is still the one
  to watch.
- Stopped five stale ephemeral tournament apps:
  `ap-7kNVf9y0HIjEVbBZv32Eew`, `ap-7ylEpRwQzYVUHe0TKG7tUW`,
  `ap-s0dosY1bwBKXuLWU3OJKFW`, `ap-q0VM2Gd38i8Ji2TlfNR4VC`,
  `ap-lSMT2MGmcnli2iTMF1Wbqi`.
- Two minutes after cleanup, the current child
  `fc-01KRTR2B7BPV74FCAY1DPMD68M` was still pending. Status had no active batch
  and had moved to `queued_waiting_for_drain` as fresh checkpoints arrived.
  Trainer refresh generation `7` appeared, but it is still sourced from
  `round-000015` / `919`, so it is not new rating progress.

## 06:30 EDT Update

- After a 5-minute sleep, `round-000031` is still active, not skipped.
- Current status:
  - latest rating remains `round-000015` / `919`;
  - intake `2027`, so `1108` checkpoints newer than latest rating;
  - active batch checkpoint/spec/roster counts all `1362`;
  - probed completed game summaries `21`;
  - newest probed game output age about `441s`;
  - operator action still says wait.
- Recent compact history now shows why the old skips happened:
  `round-000024` through `round-000030` all have
  `skip_scan_output_progress=false`, so they were skipped by old/stale
  in-flight recovery code, not by the just-deployed per-game-summary scan.
- Plain English: the current deployed fix has not repeated the bad skip yet.
  `round-000031` still needs to either keep producing output and reduce, or
  eventually publish a rating beyond `919`.

## 06:23 EDT Update

- Live recheck after the per-game-summary recovery deploy:
  - active internal game-batch artifact is `round-000031`;
  - checkpoint/spec/roster counts all `1362`;
  - probed completed game summaries: `21`;
  - newest probed output age about `78s`;
  - latest rating still `round-000015` / `919`;
  - intake `2016`, so `1097` checkpoints are newer than latest rating.
- Recent compact history confirms the previous bad pattern:
  `round-000024` through `round-000030` were skipped as
  `zero_progress_smaller_pool`.
- Plain English: the current batch is alive under the newly deployed recovery
  code. It still has not published. Next action is to wait longer and verify
  whether output keeps moving, then whether reduction publishes a rating beyond
  `919`.

## 06:22 EDT Update

- Compact recent-batch status exposed the real pattern:
  recent internal game-batch artifacts `round-000024` through `round-000030`
  were skipped as `zero_progress_smaller_pool` while latest rating stayed at
  `919`.
- That means the active-batch problem was not just display: recovery was
  repeatedly skipping smaller-than-current-desired batches instead of producing
  a newer rating snapshot.
- Found and patched the likely recovery bug:
  `_rating_round_skip_decision(..., scan_output_progress=True)` now forces
  per-game summary counting. With one Modal worker per game, relying on shard
  summaries can make real game outputs look like zero progress.
- Added regression coverage:
  a stale-looking smaller-pool batch with a fresh per-game summary is now
  `not_skippable`, with completed/started counts detected.
- Validation:
  - byte-compile passed;
  - focused skip/status tests passed: `9 passed`;
  - redeployed `curvyzero-checkpoint-tournament-v2`.
- Plain English: the latest deployed code should stop skipping live per-game
  output as zero progress. Need to recheck live and watch whether the next
  active batch publishes beyond `919`.

## 06:18 EDT Update

- After a 3-minute sleep, live status still has no rating publish beyond
  `round-000015` / `919`.
- Active internal game-batch artifact is still `round-000030`:
  - checkpoint/spec/roster counts all `1095`;
  - game count `11046`;
  - probed completed summaries `2`;
  - active batch age about `571s`;
  - newest probed game output age about `536s`;
  - status still says wait, not drain.
- Intake is now `1953`, so `1034` checkpoints are newer than latest rating.
- Plain English: the current active batch includes newer checkpoints, but it
  has not completed/reduced. It is approaching the stale threshold. Continue
  waiting/rechecking and let deployed recovery decide; do not spawn duplicate
  tournament work while status still sees this active batch.

## 06:12 EDT Update

- Added the second status hardening pass and redeployed
  `curvyzero-checkpoint-tournament-v2`.
- Active game-batch status now reports:
  - `checkpoint_count`;
  - `rating_spec_checkpoint_count`;
  - `checkpoint_roster_count`;
  - `pool_status`.
- Focused validation now covers `9` status/control tests, including spec/roster
  count reporting.
- Live status after deploy:
  - latest rating remains `round-000015` / `919`;
  - intake is `1925`, so `1006` checkpoints are newer than latest rating;
  - active internal game-batch artifact is `round-000030`;
  - active batch checkpoint/spec/roster counts are all `1095`;
  - active batch has fresh probed game output;
  - trainer refresh is generation `6`, but still sourced from `round-000015`,
    so it is not new rating progress.
- Plain English: the current active tournament batch is internally consistent
  and includes some newer checkpoints, but the leaderboard has not yet advanced
  past `919`. Keep waiting/rechecking; do not spawn duplicate tournament work
  while this batch is alive.

## 06:10 EDT Update

- Patched local/server status tooling to report `pool_status`:
  - intake checkpoint count;
  - latest published rating checkpoint count;
  - active game-batch checkpoint count;
  - whether the active game batch is larger than latest rating;
  - whether it appears to cover only the old already-rated pool.
- Focused validation passed:
  - byte-compile for the control script and tournament app;
  - `7` feedback-loop status/control tests, including the old-pool warning.
- Live status with the patched local control script now shows:
  - latest rating still `round-000015` / `919`;
  - intake `1907`, so `988` checkpoints newer than latest rating;
  - current active internal game-batch artifact `round-000030`;
  - active batch checkpoint inputs `1095`;
  - active batch is larger than latest rating, so it is not the exact `919`
    old-pool failure anymore;
  - active batch has at least `2` probed game summaries and fresh output.
- Plain English: the tournament is currently running a batch that includes some
  newer checkpoints, but ratings are still not caught up. This is still not full
  loop validation. Next proof is a published rating beyond `919`, then trainer
  refresh and trainer-proof from that newer rating.

## 06:05 EDT Update

- The newest live readout is not good enough to call success.
- After the 05:54 drain, a new active internal game-batch artifact appeared:
  `round-000029`.
- It is producing some game output, so it is not simply dead.
- But it reports only `919` checkpoint inputs, while:
  - latest published rating is still `round-000015` / `919`;
  - intake is about `1888`, so roughly `969` checkpoints are newer than latest
    rating;
  - the 05:54 drain reported it was trying to rate `1862` checkpoints.
- Plain English: the tournament is running games, but the visible active game
  batch appears to be rating only the old pool, not the newly arrived
  checkpoints. That is not proof the automatic loop is caught up.
- Current investigation: add/repair tooling so the control command explicitly
  warns when intake is ahead but the active game batch is not larger than the
  latest rating. Stop relying on raw Volume inspection or mental inference.

## 05:56 EDT Update

- Found and fixed a local operator-tooling bug:
  after a synchronous `drain-if-ready` returned, the local lease was treated as
  finished even though the detached rating call still needed time to schedule a
  game batch.
- Why it mattered: repeated `drain-if-ready` calls could have spawned duplicate
  rating calls during that scheduling window.
- Patch: a recent `drain_returned` request with a real `function_call_id` and
  no `drain_spawn_skipped_reason` now remains fresh for the lease window.
- Compact status also now includes current game-batch `updated_at` and
  `age_seconds`, so `--full-status` is not needed just to judge whether a
  zero-output batch is near the stale threshold.
- Validation:
  - `uv run python -m py_compile scripts/curvytron_live_loop_control.py`
    passed;
  - a repeated `drain-if-ready` returned `blocked_recent_drain_request`;
  - no duplicate drain spawned.
- Current pending rating call remains
  `fc-01KRTNPSZY7MCMPFMQJ9N1EAQM`.
- Next action: wait for that rating call to create a new active bounded game
  batch, or for the lease to expire if it never does.

## 05:54 EDT Update

- Stale recovery cleared `round-000028`; status became
  `ready_for_next_rating_batch`.
- Latest rating before drain remained `round-000015` / `919`.
- Intake at drain time was `1862`, so `943` checkpoints were newer than latest
  rating.
- Ran exactly one deployed drain:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action drain-if-ready --activity-probe-pairs 4 --lookahead-batches 64`
- Drain result:
  - event count `0`;
  - rating checkpoint count `1862`;
  - latest rating checkpoint count `919`;
  - desired new checkpoint count `943`;
  - stale rating claim repaired: `true`;
  - rating call id: `fc-01KRTNPSZY7MCMPFMQJ9N1EAQM`.
- Next proof: status should show a new active bounded game batch from that
  rating call. Then it must either produce game summaries and publish ratings,
  or recover cleanly without blocking the pipeline.

## 05:50 EDT Update

- Latest rating remains `round-000015` / `919`.
- Intake is now `1809`, so `890` checkpoints are newer than latest rating.
- Queue length is `113`.
- Trainer refresh remains generation `5` from `round-000015`.
- Current internal game-batch artifact `round-000028` still has `0` visible
  completed game summaries in both root progress and bounded output probe.
- Status still says `rating_game_batch_active`, not recoverable yet.
- Next action: wait a bit longer. If status changes to ready/recoverable, use
  the deployed control path; do not spawn duplicate work while status still
  sees an active batch.

## 05:49 EDT Update

- Trainer-proof for generation `5` finished.
- Summary:
  - `136` runs scanned in `9` chunks;
  - target assignment count: `24`;
  - latest applied target count: `25`;
  - assignment refresh events: `1625`;
  - assignment refresh applied rows: `532`;
  - target env rows: `53473`;
  - target provider-ok rows: `23183`;
  - target provider-false rows: `0`;
  - latest decisions: `113` applied, `21` unchanged, `2` kept previous.
- The same two runs still show `kept_previous` with `JSONDecodeError` on a
  pending assignment load:
  - `cz26a-r001-out0-n0-imm0-b20w05r1`;
  - `cz26a-r020-out100-n0-imm10-b20w05r1`.
- Plain English: generation `5` is being consumed by live trainers, but it is
  newer than generation `4`, so fewer trainers have it as their latest applied
  assignment yet (`25/136` versus `48/136` for generation `4`). Provider loads
  for target rows still show `0` false rows.
- Tooling issue fixed locally: `trainer-proof` now omits per-run rows by
  default and keeps the summary. Use `--assignment-proof-row-limit -1` only
  when full rows are needed. `py_compile` and `--help` passed after this patch.

## 05:47 EDT Update

- Latest rating remains `round-000015` / `919`.
- Intake is now `1793`, so `874` checkpoints are newer than latest rating.
- Queue length is `64`.
- Trainer refresh remains generation `5` from `round-000015`.
- Current internal game-batch artifact `round-000028` still shows:
  - `1361` checkpoint inputs;
  - `659` pairs;
  - `13839` games;
  - root completed count `0`;
  - output probe completed count `0`.
- Plain English: the current game batch exists but has not visibly produced
  game summaries yet. The deployed status still says active, not recoverable.
  Do not spawn duplicate work.
- While waiting for the stale window, `trainer-proof` CLI output was cleaned up
  so future proof checks return a compact summary by default.

## 05:44 EDT Update

- After the wait, latest rating is still unchanged: `round-000015` / `919`.
- Trainer refresh advanced again to generation `5`, but it still uses the same
  rating snapshot `round-000015`. This is a trainer-assignment rewrite, not a
  new tournament rating publish.
- Generation `5` rewrote `24` assignment pointers with new SHA prefixes.
- Intake is now `1775`, so `856` checkpoints are newer than latest rating.
- Queue length dropped to `9`.
- Current internal game-batch artifact moved to `round-000028`:
  - `1361` checkpoint inputs;
  - `659` pairs;
  - `13839` games;
  - root progress says `0` completed;
  - bounded output probe saw `0` completed game summaries so far.
- Plain English: a new current game batch exists but has not visibly written
  game summaries yet. The deployed status says to let stale recovery scan real
  output before any skip. Do not manually skip it.
- Next action: run trainer-proof for the new generation while waiting for
  `round-000028` to start writing output or become recoverable.

## 05:40 EDT Update

- Deployed status still says wait, not drain.
- Latest rating is unchanged: `round-000015` / `919`.
- Intake is now `1766`, so `847` checkpoints are newer than latest rating.
- Queue length is now `127`.
- Current internal game-batch artifact is still `round-000027`:
  - `1095` checkpoint inputs;
  - `526` pairs;
  - `11046` games;
  - root progress still says `0` completed;
  - bounded output probe found `6` completed game summaries;
  - latest probed output age is about `519s`;
  - status flags include both `active_game_batch_zero_started` and
    `active_game_batch_has_game_output`.
- Plain English: the compact root progress is stale or incomplete, but real
  game outputs exist. Do not skip or respawn from the root progress alone.
- Next action: wait and recheck. If output stops and the deployed tool says the
  batch is stale/recoverable, use the deployed recovery path. If it publishes a
  newer rating, run `refresh-if-ready` and then `trainer-proof`.

## 05:35 EDT Update

- Current state: the loop is partially proven, not fully done.
- What is now proven:
  - CZ26 shared defaults are patched and deployed;
  - trainer-facing export advanced to generation `4`;
  - generation `4` was built from rating snapshot `round-000015` with `919`
    source rows and `100` active trainer-facing rows;
  - `24` trainer assignment pointers were rewritten;
  - the read-only run-status app is deployed;
  - `trainer-proof` now checks running trainer logs through deployed functions.
- Trainer consumption proof from:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action trainer-proof --activity-probe-pairs 0 --run-limit 0`
  - scanned `136` runs in `9` chunks;
  - target assignment count: `24`;
  - latest applied target count: `48`;
  - assignment refresh events: `1563`;
  - assignment refresh applied rows: `470`;
  - target env rows: `106820`;
  - target provider-ok rows: `43723`;
  - target provider-false rows: `0`.
- Plain English: some running trainers have already loaded the generation-4
  tournament-derived assignments. It is not yet all trainers. Keep checking
  whether `latest_applied_target_count` rises above `48`.
- Two trainers reported `kept_previous` because a pending assignment load hit
  a `JSONDecodeError`. If those same rows persist on the next proof pass, debug
  the assignment file/read path.
- Current tournament/rating state from latest status:
  - latest rating is still `round-000015` / `919` rows;
  - intake has `1763` checkpoints, so `844` are newer than latest rating;
  - queue length is `93`;
  - current internal game-batch artifact is `round-000027`;
  - it has `1095` checkpoint inputs, `526` pairs, and `11046` games;
  - the output probe found game output, so it is not a zero-output dead batch;
  - status says wait for completion/reduction instead of spawning duplicate
    work.
- Next action: update docs, then wait/recheck with the deployed-function script.
  If the current game batch publishes a newer rating, run `refresh-if-ready`
  again and then rerun `trainer-proof`.

## 05:21 EDT Update

- Current state: the trainer-facing refresh leg is now fixed and proven once.
- Code/docs cleanup completed:
  - shared current-lane defaults now point at `cz26-live-20260517a` /
    `elo-cz26-live-20260517a`;
  - current GIF prefixes are `cz26a-`, `cz26b-`, and `cz26c-`;
  - current refresh config is the CZ26 config ref, not r18fresh;
  - current assignment bank is `cz26-training-candidates` /
    `try-cz26-training-candidates`;
  - current refresh pointer list has `24` CZ26 pointers.
- Focused validation passed before deploy:
  - shared current-lane/GIF tests: `10 passed, 1 skipped`;
  - loop-control/blocker tests: `4 passed`;
  - byte-compile passed for the touched scripts/contracts.
- Redeployed `curvyzero-checkpoint-tournament-v2` at about `09:19Z`.
- `refresh-if-ready` then succeeded:
  - trainer refresh advanced from generation `3` to generation `4`;
  - source rating advanced from `round-000007` to `round-000015`;
  - source row count is `919`;
  - active trainer rows are `100`;
  - rewritten refresh pointers: `24`;
  - new snapshot id: `auto-r000015-g4-ec1bef62`.
- Current compact status after refresh:
  - latest rating is still `round-000015` / `919`;
  - intake is `1660`, so `741` checkpoints are still newer than latest rating;
  - active game batch artifact is `round-000026` with `1307` checkpoint inputs,
    `632` pairs, and `13272` games;
  - status currently reports zero completed games for `round-000026`, so do not
    claim rating catch-up success yet.
- Next action: keep using status/drain tooling. Wait for `round-000026` to show
  game output or recover safely. Then prove latest rating advances beyond
  `round-000015` / `919`, and run `refresh-if-ready` again so trainer pointers
  advance beyond generation `4`.

## 05:14 EDT Update

- Current state is improved but still not fully validated.
- The tournament/rating side did move forward after the far-ahead blocker
  patch:
  - latest rating advanced from `round-000010` / `588` checkpoints to
    `round-000015` / `919` checkpoints;
  - intake had `1631` checkpoints at that check, so `712` were still newer
    than the latest rating snapshot;
  - no active game batch was visible in status at that moment.
- The next hard blocker is now the trainer-facing refresh leg, not the rating
  leg. `refresh-if-ready` failed because the deployed app defaults still point
  at the old r18fresh current-lane config:
  - old default tournament: `curvy-r18fresh-live-bounded-dsf1-20260516b`;
  - old default rating: `elo-r18fresh-live-bounded-dsf1-20260516b`;
  - old default config ref:
    `control:training/lightzero-curvytron-visual-survival/curvytron-current-control/attempts/try-curvytron-current-control/opponents/training_candidate_refresh_config.json`;
  - that config read as missing/wrong schema in the deployed refresh tick.
- Correct current CZ26 values from the launch manifest:
  - tournament: `cz26-live-20260517a`;
  - rating: `elo-cz26-live-20260517a`;
  - trainer-facing leaderboard:
    `cz26-live-20260517a-elo-cz26-live-20260517a-training`;
  - refresh config:
    `control:training/lightzero-curvytron-visual-survival/cz26-control/attempts/try-cz26-control/opponents/training_candidate_refresh_config.json`;
  - assignment bank: `cz26-training-candidates` /
    `try-cz26-training-candidates`;
  - refresh pointer count: `24`.
- Immediate patch target:
  1. update shared current-lane defaults to CZ26, not r18fresh;
  2. update tests that intentionally lock those defaults;
  3. redeploy the tournament app;
  4. run `refresh-if-ready` again and prove trainer refresh advances beyond
     generation `3` using the latest rating snapshot;
  5. continue `drain-if-ready` until ratings catch up past `919`.

## 05:01 EDT Update

- Current state is still not validated. The feedback loop is not proving that
  new checkpoints get rated and exported back to trainers yet.
- The important new root cause is simple: the live status tool said there was no
  active game batch, but app logs showed repeated rating-loop no-ops on
  `round-000024`.
- In plain English: an old internal game-batch artifact exists for the same
  tournament/rating run, but its `pool_hash`/`roster_hash` does not match the
  current desired checkpoint pool. The rating loop refuses to overwrite it,
  returns `running_existing_round`, and stops. No new useful games are created.
- Why this fooled the tooling: status only looked a fixed number of batch
  indices ahead from the latest published rating. Latest is still
  `round-000010`; the blocking artifact is `round-000024`; the lookahead was
  `12`, so status stopped at `round-000022` and falsely reported no active
  game batch.
- What this means for operators:
  - old tournaments are not the main issue;
  - the current live lane has a conflicting existing game-batch artifact;
  - raw Volume spelunking should not be required to see this;
  - status/recovery must scan for all unrated blocking artifacts, not just a
    small fixed window;
  - a zero-output different-spec artifact is safe to skip so the current
    checkpoint pool can move forward.
- Immediate patch target:
  1. make feedback-loop status discover far-ahead unrated game-batch artifacts;
  2. make drain recovery identify different-spec blockers and skip zero-output
     ones instead of letting them block forever;
  3. make the safe operator script default to a wider lookahead;
  4. redeploy, then prove latest rating advances beyond `round-000010` / `588`
     and trainer refresh advances beyond generation `3`.
- A synchronous `drain-if-ready` request started around `08:53Z` drained the
  queue but eventually hit the deployed drain function timeout (`600s`). Treat
  that as confirmation that the old deployed recovery path was still trapped
  behind the conflicting artifact, not as loop success.
- Local patch status before deploy:
  - status now scans for unrated blocking artifacts outside the fixed lookahead
    window;
  - drain recovery now identifies a different-spec existing artifact and skips
    the zero-output case instead of blocking forever;
  - the safe operator script defaults to `64` lookahead batches;
  - focused compile and regression tests passed (`11` focused tests total).
- `09:06Z`: deployed the patch to `curvyzero-checkpoint-tournament-v2`.
- `09:06Z` deployed status now reports the blocking artifacts through
  `round-000025` as skipped and says the lane is ready to rate the current
  checkpoint pool: `1626` intake checkpoints, `1038` newer than latest rating,
  latest still `round-000010` / `588`, trainer refresh still generation `3`.
- `09:06Z`: safe `drain-if-ready` spawned rating loop
  `fc-01KRTK0AMKAM34BW3FKYBAYR0C` for `1626` checkpoints and returned quickly.
  This is progress relative to the previous 600s timeout, but not loop proof.
- `09:07Z`: immediate status still showed no active new game batch. App logs
  also showed old workers from already-skipped `round-000022` still writing game
  outputs. Current interpretation: control state and old worker output can now
  disagree. Do not spawn a duplicate while the operator drain-request lease is
  fresh; give the spawned rating loop a short scheduling window, then check
  whether a new current-code batch appears.

## 04:29 EDT Update

- Current state is still not validated. The trainer -> intake -> tournament ->
  rating -> trainer-export feedback loop is stuck before rating publish.
- Safe deployed-app status at `08:29Z`:
  - active game batch: `round-000022`;
  - checkpoint inputs in that batch: `1361`;
  - pairs: `659`;
  - games: `13839`;
  - completed game summaries visible to the status probe: `0`;
  - intake checkpoint count: `1381`;
  - checkpoints not yet included in latest rating: `793`;
  - intake queue length: `85`;
  - latest rating is still `round-000010` with `588` checkpoints;
  - trainer refresh is still generation `3` from `round-000007`.
- Plain English: new checkpoints are reaching intake, but they are not yet
  turning into a newer rating snapshot, and trainers are not yet receiving a
  newer exported opponent list.
- The live control tool is the source of truth for operators:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 12`.
  Do not replace this with raw Volume browsing except when debugging the tool
  itself.
- App logs from the same window showed two timeout waves:
  - old `1800s` cancellations from pre-timeout-patch game workers;
  - newer `600s` cancellations. This means there is still a live worker timeout
    path to identify, or `round-000022` was launched on old code before the
    deployed timeout patch took effect. Do not claim the timeout issue is fixed
    until a new batch writes game summaries and reduces ratings.
- Immediate next actions:
  1. identify which function/input is still hitting `600s`;
  2. let `scripts/curvytron_live_loop_control.py --action drain-if-ready` handle
     recovery after the stale window instead of spawning duplicate work;
  3. verify the next successful batch advances latest rating beyond
     `round-000010` / `588`;
  4. verify trainer refresh advances beyond generation `3`.
- `08:32Z`: safe `drain-if-ready` saw no active game batch, queue length `90`,
  and spawned exactly one bounded drain:
  `fc-01KRTH2CJ483F5RYV945MZ4BQH`. Do not spawn another drain blindly. Next
  status check should prove whether that drain created the next active bounded
  game batch and whether game summaries start appearing.
- Tooling follow-up: `scripts/curvytron_live_loop_control.py` now writes a short
  operator drain-request lease into the intake Dict before spawning a drain. This
  is not tournament state; it is a safety guard so repeated operator calls do not
  spawn multiple drains while Modal is still scheduling the last request. Default
  lease window is `600s`.
- `08:39Z`: after several minutes the earlier drain request had not produced an
  active game batch, and queue length had grown to `482`. Re-ran
  `drain-if-ready` with the new lease guard. It spawned
  `fc-01KRTHECNW3S4EMBPZ5A3QX6JE` and recorded the lease at
  `live_loop_control:drain_request:cz26-live-20260517a:elo-cz26-live-20260517a:active`.
  Do not issue another drain request until this either creates an active game
  batch or the `600s` lease expires.
- `08:39:58Z`: verified the lease guard. A second `drain-if-ready` call returned
  `blocked_recent_drain_request` and did not spawn another drain.
- Tooling refinement after that check: future `drain-if-ready` calls now wait
  for the short deployed drain result by default and print a compact
  `drain_result_summary`. Detached drain remains available only with
  `--detach-drain`. Byte-compile passed.
- `08:42Z`: status showed the detached drain had drained the queue to `0` but
  still had not produced an active game batch. Latest rating was still
  `round-000010` / `588`.
- `08:43Z`: ran one synchronous `drain-if-ready` with
  `--ignore-drain-request-lease` to see the real drain result. It repaired a
  stale rating claim and spawned rating loop `fc-01KRTHMV0DN5TH8SBZM6EM945Y`
  over `1494` checkpoints (`906` new beyond latest rating). Next proof is an
  active game batch from this rating loop, then game summaries, then rating
  publish.
- `08:44Z`: status still showed no active game batch. Queue had only `4` new
  events, latest rating was still `round-000010` / `588`, and function-call log
  lookup for `fc-01KRTHMV0DN5TH8SBZM6EM945Y` returned no log lines. Current
  interpretation: the rating-loop call may be waiting to schedule, or it failed
  before writing logs. Keep monitoring before declaring another code bug.
- Scheduled-path cleanup: patched and deployed
  `curvytron_checkpoint_intake_drain_tick` so it computes feedback-loop status
  and uses the same `drain-if-ready` decision before spawning drain work. This
  should stop the scheduled path from blindly spawning drains every scan while
  an active/stale batch or recent request is already in play. Focused control
  tests passed (`2 passed`), byte-compile passed, deploy completed.
- `08:49Z` post-deploy status: still no active game batch; latest rating still
  `round-000010` / `588`; intake has `1496` checkpoints, `908` not in latest
  rating, and queue length `12`. Pending drain request lease for
  `fc-01KRTHMV0DN5TH8SBZM6EM945Y` remains active until about `08:53Z`, so do not
  spawn again before then unless status changes.
- Post-deploy system logs still show old function-id `600s` cancellations. These
  are likely pre-deploy scheduled tick inputs aging out, not proof that the new
  gated tick is failing. Keep watching after the old 10-minute window clears.
- Added regression coverage for the gated scheduled tick: it spawns only when
  the feedback-loop gate allows it, and it blocks when an active game batch is
  present. Focused tests passed (`4 passed`), then redeployed the app again.

## 03:50 EDT Update

- Current live loop is still not fully proven. Do not claim success until a
  current-code bounded game batch completes, writes a newer rating snapshot,
  advances the trainer-facing export, and trainers load that export.
- The immediate blocker was not "old tournaments." The current lane had stale
  live control state and old in-flight code paths that could skip active
  game-batch artifacts without scanning real game outputs.
- Deployed fixes now in local code and tested:
  - skipped game-batch summaries expose the actual skip decision;
  - CLI detached `intake-drain` uses `.spawn()` instead of `.remote()`;
  - `loop-status` reports multiple active game-batch artifacts and stale output;
  - `latest.json` publication is monotonic and cannot be overwritten by an
    older or smaller snapshot;
  - scheduled drain tick now spawns the drain function instead of mutating
    tournament state inline from a temporary/stopping local app.
- New operator tooling is being added: `--mode loop-control`. It asks the
  deployed status function for the truth, decides whether a drain is safe, and
  can spawn the next bounded drain without raw Volume inspection.
- Focused local validation for the first `loop-control` decision tests passed:
  `5 passed`.
- Important tooling correction: do not use `modal run` for ordinary live
  status/control anymore. It creates a temporary Modal app, and scheduled
  functions on that temporary app can fire while the app is stopping. That is
  how we got `ConflictError: The app is stopped or disabled` from a spawned
  drain. Use `scripts/curvytron_live_loop_control.py`, which calls deployed
  functions by name and does not create a scheduled temporary app.
- Deployed-app control read at `08:01Z`: status is `rating_game_batch_active`.
  Current active game batch is `round-000020`, `1228` checkpoint inputs, `593`
  pairs, `12453` games, bounded config `adaptive_v0 / 300 / 100`, and the
  activity probe found `21` completed game summaries with output age about
  `84s`. Latest rating is still the regressed `round-000010` / `588`
  checkpoint snapshot. Trainer refresh is still generation `3`.
- `round-000019` was skipped by the bad transient path with
  `scan_output_progress=false`. Treat it as evidence that `modal run` live
  control was the wrong tool, not as evidence that deployed output scanning is
  broken.
- Current next action: wait for `round-000020` to finish/reduce, then verify
  latest rating advances beyond `round-000010` / `588` and trainer refresh
  advances beyond generation `3`.
- After another wait, deployed-app status at `08:05Z` and `08:07Z` still shows
  `round-000020` active. The probe still finds only one completed pair
  (`21` game summaries), and latest sampled output aged from about `345s` to
  `497s`. This is not success yet. It means the batch has some real output,
  but it is not visibly progressing in the sampled/full pair-only scan.
- Patched and redeployed two cleanup items from the critique lane: the deployed
  `operator_next_action` strings now point at
  `scripts/curvytron_live_loop_control.py`, and unsafe local-entrypoint live
  modes (`loop-status`, `loop-control`, `intake-drain`) now raise with a clear
  message instead of quietly encouraging the old temporary-app path.
- App logs showed the deeper reason active game batches can stall: tournament
  game inputs are hitting the Modal worker timeout of `1800s` and getting
  canceled/rescheduled. That is a hidden cap even though the game `max_steps`
  is `1048576`. Local patch now raises tournament game, shard, pair, and rating
  round function timeouts to `24h`. Existing `round-000020` workers were
  launched before this patch, so expect that batch to either finish from old
  workers or get skipped by recovery; the next spawned batch should use the
  longer timeout after deploy.
- Deployed the `24h` timeout patch. `drain-if-ready` at `08:11Z` correctly
  refused to spawn a duplicate because the live loop had already advanced to
  active `round-000021`. `round-000021` has `1181` checkpoint inputs, `569`
  pairs, `11949` games, and the activity probe found `15` completed summaries
  with output age about `106s`. Latest rating is still `round-000010` / `588`,
  so the loop remains unproven until a current batch writes ratings.
- Follow-up at `08:15Z`: `round-000021` still active, but still only `15`
  probe-visible summaries and latest output age about `325s`. This suggests
  the current batch is also not visibly progressing yet. Next check is after
  the `600s` stale window; `drain-if-ready` should either block because the
  batch has fresh progress or let recovery move to a next batch.
- After the stale-window wait, `drain-if-ready` at `08:23Z` did not spawn a
  duplicate, because recovery/scheduling had already advanced to active
  `round-000022`. `round-000022` has `1361` checkpoint inputs, `659` pairs,
  `13839` games, and no output yet. This is the next batch that should reflect
  the deployed `24h` worker timeout. Latest rating remains `round-000010` /
  `588`.

## 03:10 EDT Update

- Important correction: the loop is still not fully validated.
- Live `loop-status` at `07:04Z` showed `round-000015` was alive: intake
  `956`, latest rating `414`, active game batch `919` checkpoint inputs, and
  the bounded activity probe found `21` completed game summaries.
- Follow-up `loop-status` at about `07:07Z` showed no current game batch and
  `round-000015` marked `skipped` at `07:06:59Z`, while latest rating was
  still `414`. That is a contradiction, not success.
- Current truth: either an older deployed drain call skipped the batch, or the
  full recovery scan still missed output under live conditions. Do not assume
  the safety fix worked until the next batch proves it.
- Tooling patch: `loop-status` now exposes the skipped
  game-batch `skip_decision` summary, including completed game count, latest
  result timestamp, stale age, scan mode, and scan errors. Operators should
  see why a batch was skipped without raw Volume spelunking.
- Focused local validation for this patch passed: `39 passed`.
- Deployed the skip-decision `loop-status` patch to
  `curvyzero-checkpoint-tournament-v2`.
- Detached drain attempt exposed another control-plane bug: CLI
  `intake-drain` with `spawn_rating=true` used `.remote()` even though Modal
  warns detached `.remote()` calls may be canceled when the local caller exits.
  Patched this path to use `.spawn()` and return a function-call id instead.
  Focused local validation stayed green: `39 passed`.
- Deployed the `.spawn()` drain fix to `curvyzero-checkpoint-tournament-v2`.
- Fast live `loop-status` with the activity probe disabled returned at
  `07:21Z`. Current state: latest rating advanced to `round-000011` with
  `681` checkpoints, active game batch is `round-000017` with `960` checkpoint
  inputs, intake has `1063` checkpoints, queue length is `83`, and trainer
  refresh is still old generation `3` from `round-000007`.
- The skipped-batch summaries now explain the old bad behavior: skipped
  `round-000013` through `round-000016` all show
  `scan_output_progress=false`. Those skips came from old/stale control-plane
  code and are not proof that the current safety path is broken.
- Next proof target: `round-000017` must write game outputs, reduce into a new
  rating snapshot, advance the trainer-facing export beyond generation `3`, and
  be loaded by trainers.
- Follow-up `loop-status` at `07:22Z` with a 4-pair activity probe found real
  output for `round-000017`: `21` completed game summaries from `1` sampled
  pair, no scan error, latest result timestamp present. Current action is
  wait for that bounded game batch to finish or reduce; do not spawn another
  competing game batch while this one is alive.
- Follow-up after a 3-minute wait at `07:26Z`: `round-000017` still has real
  output but has not reduced yet. Status also showed another un-rated artifact,
  `round-000016`, still present as `running/input_written`. Patched
  `loop-status` to report `active_game_batch_count` and flag
  `multiple_active_game_batches` so this cannot be hidden. Focused validation:
  `40 passed`.
- Patched `loop-status` again so the activity probe reports
  `latest_result_age_seconds`, `stale_after_seconds`, and
  `active_game_batch_output_stale`. This distinguishes "some output exists"
  from "fresh output is still landing." Focused validation: `41 passed`.
- Live `loop-status` after that deploy at `07:30Z`: no active game batch.
  `round-000017` had been skipped, and its `skip_decision` also showed
  `scan_output_progress=false`, so it was skipped by stale in-flight code.
  Latest rating unexpectedly read as `round-000010` / `588` checkpoints even
  though `round-000011` / `681` had already been observed. That suggests an
  older reducer can overwrite `latest.json`; this now needs a monotonic latest
  write guard.
- Started a new detached intake drain with the fixed `.spawn()` path:
  `fc-01KRTDHBHZH0KWC4M5A97WCWNA`.
- Found and patched the likely `latest.json` regression: child reduction had a
  monotonic latest guard, but the parent `rating_loop` rewrote `latest.json`
  unconditionally after each child returned. Latest publishing is now
  centralized and blocked when the candidate is older, explicitly marked
  non-global, or has fewer checkpoint rows than current latest. Focused
  validation: `44 passed`.
- Deployed the monotonic latest guard.
- Live `loop-status` at `07:36Z`: detached drain created active game batch
  `round-000018` with `1095` checkpoint inputs, `526` pairs, and `11046`
  games. The activity probe found `21` completed game summaries with output
  age about `61s`, no scan error. Latest is still the regressed
  `round-000010` / `588` checkpoint snapshot until a newer current-code
  reduction publishes over it.
- After a 5-minute wait, `round-000018` was also skipped with
  `scan_output_progress=false`, so stale in-flight scheduled code was still
  mutating progress. The status run also surfaced a
  `curvytron_checkpoint_intake_drain_tick` error: it ran drain logic via
  `.local()` and then tried to spawn rating work from a stopped/disabled app.
- Patched scheduled drain tick so it only spawns a drain function call and does
  not run destructive recovery/skipping inline. Focused validation: `45
  passed`.

## 02:41 EDT Update

- Corrected live mental model: a zero-started compact progress file is not
  proof that no Modal game workers are running. App logs showed `r000012` game
  outputs completing after the control plane had treated that game batch as
  stale.
- Root issue: stale recovery was checking the game-batch input timestamp and
  coarse progress file, but the drain path was not scanning real game outputs
  before deciding to skip. The helper already supported output scanning; the
  live recovery path did not use it.
- Current patch: intake drain now calls stale game-batch recovery with
  `scan_output_progress=True`. The intended rule is simple: only skip an
  apparently stuck game batch if there is no recent real game output. If the
  scan itself fails, do not skip silently.
- Focused local validation for this patch passed: `36 passed`.
- Deployed patched `curvyzero-checkpoint-tournament-v2` after the focused
  validation.
- This means the earlier "recover after 600s" wording was incomplete. The
  correct rule is: after `600s`, scan real output; if games are still finishing,
  keep waiting or reduce when complete; if no real output exists, skip and move
  on.
- `round-000013` was spawned with `815` checkpoint inputs after the
  continuation-pool fix. That proves the pool-regression patch worked, but it
  does not yet prove the batch completed or produced the next rating snapshot.
- Live `loop-status` at `06:47Z`: intake had `828` checkpoints and latest
  ratings still had `414`. `round-000013` had been skipped, and the active
  game batch was `round-000014` with `681` checkpoint inputs. Modal logs showed
  `r000014` games actively completing, so it must not be skipped just because
  compact progress says zero-started.
- Next tooling patch: `loop-status` should include the same real-output scan
  used by recovery, so operators do not have to read logs just to tell whether
  an apparently zero-started game batch is actually alive.
- Local validation after adding the `loop-status` output scan passed:
  `37 passed`.
- Deployed the updated `loop-status` output scan.
- The first deployed output-scan status call was too slow because it still
  scanned too much. Split the paths: recovery keeps the full safety scan before
  skipping; `loop-status` now uses a bounded activity probe over expected
  active-batch battle directories. Focused local validation stayed green:
  `38 passed`.
- Deployed the bounded `loop-status` activity probe.
- The first bounded probe still sampled too much for a fast operator check.
  Tightened `loop-status` to sample fewer pairs and stop after the first real
  game output. Focused local validation stayed green: `38 passed`.
- Deployed the lighter `loop-status` activity probe.
- Live `loop-status` at `07:04Z` is now fast enough and directly reports active
  output. Current state: intake `956`, latest rating `414`, active game batch
  `round-000015` with `919` checkpoint inputs, `438` pairs, `9,198` games.
  The bounded activity probe found `21` completed game summaries from one
  sampled pair, so this game batch is alive and should not be skipped.
- Remaining gap: latest rating and trainer refresh are still behind. Need
  `round-000015` to complete or reduce, then trainer-facing leaderboard/export
  must advance beyond generation `3`.

## 01:55 EDT Update

- Historical note: first-line status tooling was previously `--mode
  loop-status` through `modal run`. That is now superseded for live control by
  `scripts/curvytron_live_loop_control.py`, which calls the deployed app by
  name and does not create a temporary scheduled app. Raw artifact reads remain
  deep forensics, not the normal operator path.
- Focused local validation for this patch passed: `34 passed`.
- Duplicate tournament calls with an existing active game-batch input now back
  off instead of crashing or overwriting. The returned status is
  `running_existing_round` with phase `existing_input_different_spec`.
- Game-batch stale recovery is separate from intake claim stale recovery:
  intake claims stay conservative at `24h`; an apparently stuck tournament game
  batch uses a `600s` check, but must scan real game outputs before skipping.
- Bounded tournament game batch `round-000006` completed: `300` pairs,
  `6,300` games, `0` failed games, ratings written.
- Training-candidate auto-refresh then succeeded from a newer rating snapshot:
  source `round-000007`, `414` rows, `100` active rows, generation `3`,
  snapshot `auto-r000007-g3-b2b6f2bc`, `24` refresh pointers rewritten.
- Intake has since seen `545` checkpoints. The latest rating snapshot used by
  trainer refresh had `414` checkpoints, so about `131` newer checkpoints still
  need tournament coverage and rating reduction before they can affect trainer
  assignments.
- Live `loop-status` at `05:58Z` showed internal game-batch artifact
  `round-000008` as zero-started, then a later `loop-status` at `06:02Z` showed
  `round-000008` complete: `300` pairs, `6,300` games, ratings written.
- Current concern moved to `round-000009`: it has `414` checkpoint inputs,
  `300` pairs, `6,300` games, and was zero-started at `06:02Z`. It was only
  about two minutes old then, so the correct behavior is to let it start or let
  the `600s` recovery window skip it if it never starts.
- Live `loop-status` at `06:11Z` showed `round-000009` complete, then
  `round-000010` skipped by stale recovery after staying zero-started, and
  `round-000011` spawned with `681` checkpoint inputs. Queue length was `0`.
  This proves the zero-started game-batch guardrail moves the live pipeline
  forward instead of blocking forever.
- Live `loop-status` at `06:26Z` exposed a second bug: after `round-000011`
  was skipped, the next game batch `round-000012` regressed to `414`
  checkpoint inputs even though intake had `705`. Root cause: continuation
  code preferred stale `seen_checkpoint_refs` over the current manifest
  `checkpoint_refs`. This is now patched so continuation uses the current
  manifest pool plus any seen extras. Focused tests passed again: `34 passed`.

## Names

- Full training batch: `cz26-full-20260517a`.
- Live tournament: `cz26-live-20260517a`.
- Live rating run: `elo-cz26-live-20260517a`.
- Training leaderboard:
  `cz26-live-20260517a-elo-cz26-live-20260517a-training`.
- `round-000003`, `round-000006`, etc. are only internal tournament
  game-batch directory names. They are not training rounds. Trainers keep
  running continuously and keep writing checkpoints.

## Validated

- The full batch is launched: `136` training runs.
- Seed tournament path completed: `4` checkpoints, `6` pairs, `126` games,
  `0` failed games.
- Seed leaderboard publication and training-candidate refresh path worked from
  the seed snapshot.
- Live repair is deployed for the checkpoint tournament app.
- The active cz26 manifest is now bounded live scheduling:
  `pair_selection=adaptive_v0`, `pairs_per_round=300`,
  `active_pool_limit=100`.
- Checkpoint intake is live and has seen `431+` checkpoints from the cz26 runs.
- Checkpoint intake later reached `545` checkpoints.
- The old live all-pairs game batch was not useful: it tried to schedule about
  `103,950` games in one shot and stalled. It has been skipped as stale.
- Bounded tournament game batch `round-000006` completed: `300` pairs,
  `6,300` games, `save_gif=false`, `0` failed games, ratings written.
- Trainer-facing refresh succeeded after the bounded tournament path:
  generation `3`, `100` active rows, `24` refresh pointers rewritten.
- The root `progress.json` can lag while game workers are running. Treat direct
  game logs and later `latest.json`/ratings output as the completion proof.
- Do not use the aggressive `claim_stale_after_seconds=60` recovery knob on the
  currently running bounded batch. The deployed default is `24h`; the 60-second
  value was only for clearing old dead batches.

## Important Semantics

- Every discovered checkpoint should enter the tournament pool.
- That does not mean the system should block on one giant "everyone plays
  everyone" batch before updating anything. For a live stream of hundreds of
  checkpoints, the robust behavior is bounded continuous game batches.
- The tournament leaderboard is updated after a game batch reduces into
  ratings. Trainers then consume the trainer-facing leaderboard export through
  opponent-refresh assignments.

## Still To Validate

- Re-run compact `loop-status` after `round-000013` has had time to start,
  complete, or recover. It should continue from the current intake pool, not
  regress to `414`.
- Verify stale recovery no longer skips a game batch while workers are still
  writing game summaries.
- Verify trainer pickup after generation `3`: env/provider rows should show the
  refreshed assignment SHA after the next refresh boundary.
- Continue watching new checkpoints past `681` and prove they enter later
  bounded game batches, reach ratings, and refresh trainer assignments.

## Next Checks

- Run:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 12`
- If `loop-status` reports an active zero-started game batch older than `600s`,
  recovery must scan real game outputs first. If games are still writing, let
  them finish; if no real output exists, skip and move on.
- Compare leaderboard generation, assignment SHA, and trainer provider-load
  rows to prove trainer consumption.
