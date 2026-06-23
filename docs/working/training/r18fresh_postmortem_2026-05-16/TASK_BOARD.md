# Task Board

Last updated: 2026-05-18 10:24 EDT.

This is the small tactical board. Keep it shorter than the analysis docs.
For the latest cz26 operator truth, start with
`CZ26_CURRENT_TRUTH_2026-05-17.md`.

## Active And Ready Lanes

| ID | Priority | Lane | Owner | Status | Question | Next Action | Artifact Predicate | Integration Target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OBS-1 | P0 | lineage | main or worker | ready | Can we prove every checkpoint crossed each pipeline boundary? | Implement best-effort `lineage_events.jsonl` boundary events or an equivalent lineage command. | One checkpoint has ordered events from `checkpoint_written` through `trainer_assignment_applied`. | `OBSERVABILITY_PLAN.md`, `TESTING_AND_GAPS.md` |
| OBS-2 | P0 | tournament health | main | patched-batch-active | Can an operator tell whether checkpoints are missing, waiting, playing, active, retired, or exported? | Wait/recheck `round-000043`; require it to publish ratings/export next. | `round-000043` is active with `4512` refs, `300` pairs / `6300` games, queue `0`, and fresh game output; it covers current intake and is newer than latest rating. | `SIGNALS.md`, tournament UI/API, `WHAT_THE_HELL_IS_GOING_ON_HANDOFF_2026-05-17.md` |
| OBS-3 | P0 | trainer consumption | main or worker | partially-proven | Are tournament winners actually becoming frozen opponents in trainers? | Re-run `trainer-proof` after next export boundary; require latest-applied target count to continue increasing and provider-false = 0. | `trainer-proof`: `46` latest-applied target count, latest decisions `37` applied / `99` unchanged, `0` provider-false rows. | `SIGNALS.md`, `TESTING_AND_GAPS.md` |
| OBS-4 | P0 | trainer health | main | running | Are the launched Grid A/Grid B trainers alive and producing checkpoints? | Continue monitoring trainer fast status; distinguish trainer artifact count from tournament-visible unique refs. | Fresh fast status: `136/136` rows, `92` completed, `44` running, `0` failed, checkpoint artifact sum `7746`, latest iteration max `326739`. | `CZ26_CURRENT_TRUTH_2026-05-17.md`, `TESTING_AND_GAPS.md` |
| POL-1 | P0 | policy observation | main | active | Do training and tournament eval expose the same controlled-player view while randomizing physical role? | Add a direct tournament regression test and link the general contract. | Training fixed/random seats and tournament balanced seats all prove controlled-player view. | `curvytron_feedback_loop/POLICY_OBSERVATION_CONTRACT.md`, tests |
| ANA-1 | P1 | r18fresh analysis | main | active | Which settings actually produced useful signal? | Finish deduped top candidates and survival/tournament comparison. | Top candidates include tournament-best, survival-best, own-reward-best, deduped by run/setting. | `FINDINGS.md`, `NEXT_BATCH_SEEDING.md` |
| ANA-2 | P1 | own-latest control | main or explorer | waiting_for_artifact | Does own-latest show the same retention failure without tournament feedback? | Re-pull after more checkpoints. | Control has enough matched eval points to compare first/best/latest against selected r18fresh rows. | `TREND_ANALYSIS.md`, `FINDINGS.md` |
| OPS-1 | P1 | asset hygiene | main | cleaned-active-apps | What is current, preserved, archive, or cleanup candidate? | Keep `ASSET_REGISTRY.md` current before kill/launch/sleep. | Every live app/arena/rating/run prefix has status and preserve rule. | `ASSET_REGISTRY.md` |
| OPS-2 | P1 | naming hygiene | main | active | Are operator-visible names concise and meaningful? | Apply `NAMING_CONVENTIONS.md` to manifest builders, canaries, tournament ids, rating ids, and GIF prefixes. | New current-code artifacts use `cz26*`, `out*`, `n*`, `imm*`, and short recipe codes while preserving full structured settings. | `NAMING_CONVENTIONS.md`, `NEXT_BATCH_DESIGN.md`, manifest tests |
| TEST-1 | P1 | synthetic full loop | worker | ready | Can a local test close fake checkpoint -> intake -> rating -> export -> refresh -> trainer consumption? | Add a synthetic test after lineage events exist. | Test asserts ordered stages and same assignment SHA across export and trainer-load rows. | `TESTING_AND_GAPS.md` |
| OPS-3 | P1 | batch ownership | main or worker | ready | Can we inspect/cancel the exact live compute owner for a batch? | Persist drain/rating/reduce call ids per active batch and expose them in status. | Status shows owner call ids for the current active batch without log guessing. | `TOURNAMENT_STABILITY_HANDOFF_2026-05-17.md` |
| EXP-1 | P2 | next batch design | main | ready | What should the next batch test after r18fresh? | Wait for ANA-1/ANA-2/OBS gates, then choose small matched panel or larger sweep. | Design names fixed axes, changed knobs, controls, launch gate, and cleanup rule. | `NEXT_BATCH_DESIGN.md` |

## Latest Reorientation Check

2026-05-18 10:24 EDT:

- Added `CZ26_BATCH_RATIONALE_REORIENTATION_2026-05-18.md` as the current
  source-of-truth note for why `cz26-full-20260517a` exists.
- Plain answer: `cz26` was launched because `r18fresh` showed real mid-run
  learning and tournament strength, but the apparent winner was confounded.
  Grid A tests robustness across reward/noise/leaderboard immortality for four
  mixed recipes. Grid B tests slot population directly, including pure blank,
  pure wall, and pure rank1 controls.
- The three analysis signals remain separate:
  reward progression, survival progression, and tournament performance.
  Reward is diagnostic, especially because tournament-fed opponents can get
  stronger over time.
- Exact manifest truth remains `136` rows: `96` Grid A, `40` Grid B, no canary
  rows inside the full manifest. The `cz26c` canary is separate.

2026-05-17 21:01 EDT:

- After local restart, live Modal state is still intact.
- Full `cz26-full-20260517a` trainer batch remains accounted for:
  `136/136` rows, `92` completed, `44` running, `0` failed.
- Trainer checkpoint artifact sum is now `7746`; latest iteration max is
  `326739`.
- Tournament remains on bounded active game batch `round-000043`:
  `4512` refs, `300` pairs / `6300` games, queue `0`.
- Game output for `round-000043` is fresh. Latest completed rating is still
  `round-000040` / `4192`, so the remaining proof is rating publish, trainer
  export refresh, and trainer consumption of the newer export.
- Fresh `trainer-proof` scan completed: `46` latest-applied target count,
  latest decisions `37` applied / `99` unchanged, and provider-false `0`.
- Follow-up tournament liveness status still has no blockers. The active batch
  has `4512` refs, while `17` newer refs are queued for the next bounded batch.

2026-05-17 20:31 EDT:

- The full `cz26-full-20260517a` trainer batch is running/completing:
  `136/136` rows accounted for, `131` running, `5` completed, `0` failed.
- The tournament latest rating is `round-000040` over `4192` checkpoint refs.
- Trainer export is current with that rating: generation `2` from
  `round-000040`.
- Active tournament game batch `round-000041` is not adding new information:
  it repeats the same `4192` checkpoint pool while queue length is `0`.
- Root cause found: `spawn_if_existing` incorrectly implied `spawn_if_empty`.
  Local code now requires explicit `spawn_if_empty=True` for an empty same-pool
  rerate. Focused tests pass.
- Next gate: deploy the tournament patch, clear/skip current useless active
  batch if needed, then prove the next fresh checkpoint intake schedules one
  bounded batch and flows back into trainer assignments.

2026-05-17 20:36 EDT:

- The same-pool rerate guard is deployed to
  `curvyzero-checkpoint-tournament-v2`.
- Recovery drain inspected active `round-000041` and did not skip it because it
  found `1431 / 6300` completed games and recent output. It is wasteful, not
  dead.
- Trainer feedback proof improved: `32` trainers applied latest target
  assignment SHAs from the `round-000040` export; provider-false count is `0`.
- Next gate: wait/recheck the active batch and the next checkpoint intake cycle.
  The expected behavior is no further empty same-pool batch; new checkpoints
  should form the next bounded useful batch.

2026-05-17 20:42 EDT:

- New checkpoint intake/drain worked after deploy:
  - intake pool `4192 -> 4462`;
  - `270` refs newer than latest rating;
  - stale same-pool `round-000041` skipped;
  - new rating call spawned:
    `fc-01KRW8GE5GRD2M3J0TF6G9YWYB`.
- Trainer fast status remains healthy: `82` running, `54` completed,
  `0` failed.
- Next gate: verify the new rating call materializes a bounded game batch or
  ratings output, then prove the resulting trainer export is consumed.

2026-05-17 20:44 EDT:

- New rating call materialized into `round-000042`.
- `round-000042` covers the full current intake pool: `4462` refs.
- It is newer than the latest completed rating by `270` refs.
- It is bounded: `300` pairs / `6300` games.
- Queue is drained to `0`.
- Next gate: game output, then ratings/latest, then export refresh and trainer
  consumption.

2026-05-17 20:51 EDT:

- `round-000042` exposed a second bug: it was skipped because newer checkpoints
  arrived after it started. That is wrong.
- Patched rule: later arrivals are backlog for the next batch; they do not
  invalidate a running batch unless the batch is stale or already no newer than
  latest rating.
- Redeployed tournament app with the fix.
- Tests:
  - focused newer-zero-output skip regression passed;
  - intake repair suite passed (`37` tests);
  - skip/drain subset passed (`8` tests).
- New patched rating call:
  `fc-01KRW9208TSJDJ8S7NR0BAF6G9`.
- Next gate: verify the new call becomes a bounded game batch and remains
  running while more checkpoints arrive.

2026-05-17 20:53 EDT:

- Patched call `fc-01KRW9208TSJDJ8S7NR0BAF6G9` materialized into
  `round-000043`.
- `round-000043`: `4512` refs, `300` pairs / `6300` games, queue `0`.
- The batch is newer than latest rating and covers current intake.
- Next gate: verify it remains active and writes game output after a quiet
  interval.

2026-05-17 07:10 EDT:

- Current deployed-function status shows internal game-batch artifact
  `round-000033` active.
- It includes `2192` checkpoint inputs, `2192` rating-spec checkpoints,
  `2192` roster rows, `1075` pairs, and `22575` games.
- It covers the current intake pool; missing-from-intake count is `0`.
- Latest published rating is still `round-000015` / `919`, so `1273`
  checkpoints are newer than the latest rating.
- Trainer refresh generation `7` still sources `round-000015`; not new
  tournament progress.
- At age about `57s`, no completed output was visible yet. Correct action:
  wait/recheck with compact status. No duplicate drain.
- Operator tooling cleanup: `curvytron_live_loop_control.py` now keeps the
  default pending-call probe compact by reporting call-graph counts and a small
  sample. Full nested call graphs are opt-in with `--full-drain-call-graph`.
- Same operator tooling pass added `proof_chain` and `blockers` to compact
  status so the output plainly shows which feedback-loop boundary is proven or
  stale.
- 07:15 recheck: `round-000033` is alive. Probe saw `21` completed game
  summaries, newest about `11s` old; Modal call summary saw `44` tournament
  game calls succeeded and `52` pending. Latest rating is still `919`, so the
  next action is wait/recheck, not duplicate drain. `60` queued checkpoints
  arrived after this active batch started.
- 07:28 recheck: `round-000033` is still active, latest rating still `919`,
  intake now `2308`, queue `116`. Probe still sees `21` completed game
  summaries, newest about `203s` old. Compact call graph sample shows `45`
  tournament game successes and `51` pending. Status reports no hard blockers;
  continue waiting for remaining games/reduce.
- 07:44 recheck: `round-000033` is still active and not stale. Latest probed
  output age is about `19s`; latest rating remains `919`; intake is `2406`;
  queue is `214`. Trainer refresh advanced to generation `8` but still sources
  `round-000015`, so it is not new tournament progress.
- 08:15 recheck: `round-000033` is still active, not published. Sampled Modal
  graph progressed to `67` tournament game successes and `29` pending.
  Provisional loop is `SUCCESS`, rating round is still pending. Latest probed
  output age is about `599s`, near the stale threshold; recheck sooner.
- 08:20 recheck: narrow output probe is stale (`944s`), but call-graph sample
  still progressed to `75` tournament game successes and `21` pending. Treat as
  slow long-running games, not stuck yet. Recheck on short cadence.
- 08:26 recheck: sampled Modal call graph now shows all `96` tournament game
  children as `SUCCESS`; only rating loop/rating round remain pending. Latest
  rating is still `919`, so next gate is rating reduction/publish.
- 08:31 log check: app logs show `round-000033` game workers still finishing at
  `12:30-12:31Z`, with runner disappearance/reschedule noise. The active issue
  is game throughput/retries, not intake. Do not spawn duplicate drains.
- 08:39 recheck: latest rating still `round-000015` / `919`; active
  `round-000033` still has `2192` checkpoint inputs, `1075` pairs, and
  `22575` intended games. Intake is now `2668` with queue `476`. App logs show
  fresh game completions at `12:36-12:39Z`, so the batch is alive. Critical
  tooling caveat: `probe_completed_game_count=21` is only a liveness sample,
  `completed_game_count=0` is the unreduced root summary, and the Modal graph is
  capped/sample-like. Next action is status/progress tooling cleanup plus
  continued monitoring, not a duplicate drain.
- Local status/progress tooling patch is ready and focused status tests passed.
  It separates root summary counts, liveness-sample counts, and progress-probe
  counts in both deployed status payloads and the local control script. It still
  needs intentional deployment before live status output uses the new fields.
- 08:52 proof: `round-000033` published as latest rating with `2192`
  checkpoints and `22575` completed games. `refresh-if-ready` wrote generation
  `11`, snapshot `auto-r000033-g11-50405322`, from `round-000033`, with `100`
  active rows and `24` rewritten pointers. `trainer-proof` scanned `136` runs:
  `3` already have a generation-11 target as latest-applied; generation-11
  provider rows are `4548 ok`, `0 false`, and `2115 null/pending`. The core
  feedback path has happened once; full catch-up is still in progress. There
  are `522` newer queued checkpoints waiting for the next bounded drain.
- Spawned exactly one next bounded drain for those `522` queued checkpoints:
  `fc-01KRV02A38VYDJPE3JWM4A0W4M`. Next proof is active game-batch creation,
  then publish beyond `2192`.
- Follow-up: that drain returned and spawned rating loop
  `fc-01KRV02Z7H8Q7466J50EPSPZ7E`; queue is now `0`; no active game batch is
  visible yet. Logs say the rating loop is waiting for CPU worker capacity.
  Wait/recheck. Do not spawn a duplicate drain while that call is pending.
- 09:08 recheck: after stopping/redeploying the crowded app state, a fresh drain
  `fc-01KRV0JK89X9FFFYHHCZEMFNBJ` spawned rating loop
  `fc-01KRV0JQYM98FDTZD6Q55VY04B`, and active game-batch artifact
  `round-000034` is now running. It covers `2739` checkpoints, `1348` pairs,
  and `28308` games. Liveness probe saw fresh game output. Latest rating is
  still `round-000033` / `2192` until this batch publishes.
- 09:25 recheck: active `round-000034` still running. Latest rating remains
  `round-000033` / `2192`; intake is `2877`; active batch covers `2739`; queue
  is `72`. Liveness sample saw `21` completed summaries, newest about `157s`
  old. Trainer fast summary says `136/136` heartbeats/progress files, `134`
  running, `2` failed, checkpoint range `8..36`, total checkpoint artifacts
  `2910`.
- Failed row detail: `cz26a-r013...` failed after `iteration_170000` with `36`
  checkpoints; `cz26b-r028...` failed after `iteration_30000` with `8`
  checkpoints. Heartbeat/status fast fields do not expose a failure reason.
- Grid B contains pure slot controls `b100`, `w100`, and `r1` inside the main
  batch. Separate `cz26c` canary status: completed, `350` checkpoints, latest
  progress iteration `17385`.
- 09:47 follow-up: `round-000034` still active, latest rating still
  `round-000033` / `2192`, intake `2989`, queue `184`. Generation-12 trainer
  export still sources `round-000033`, but `trainer-proof` is strong:
  `62/136` latest-applied target count, `48666` provider-ok rows, `0`
  provider-false rows.
- 10:19 follow-up: `round-000034` still active after about `60m`; latest
  sampled output age about `341s`, below stale threshold. Latest rating still
  `round-000033` / `2192`; intake `3136`; queue `331`. Generation `13` still
  sources `round-000033`.
- 10:30 follow-up: launch and trainer consumption are solid; tournament publish
  is still the gap. Fast trainer status saw `136/136` heartbeats/progress
  files, `134` running, `2` failed, and `3185` checkpoint artifacts. Current
  intake is `3193`; latest rating is still `round-000033` / `2192`; active
  `round-000034` covers `2739` checkpoints, `1348` pairs, and `28308` games.
  App logs show successful games at 10:29 EDT, and a wider liveness probe saw
  output about `44s` old. `trainer-proof` saw `102/136` latest-applied target
  count, `74170` provider-ok rows, and `0` provider-false rows for generation
  `13`. Expensive full progress probe timed out at `300s`; do not claim a total
  completion percentage until that tooling is fixed.
- 11:20 fix/recovery: patched and deployed the actual control-plane issue and
  future batch-size issue. Stale active batches can now enter recovery; scheduled
  drain tick probes activity; adaptive `pairs_per_round` is now a hard cap; big
  recovery scans are bounded; recovery output exposes the skip decision.
  Focused tests passed. Live recovery did not skip `round-000034` because it
  found fresh real activity: `5103 / 28308` games complete in bounded scan,
  `243 / 1348` started pairs, root progress visible at `4414 / 28308`. Current
  batch is slow/oversized, not dead.
- 11:23 status: latest rating is still `round-000033` / `2192`. Intake is
  `3427`; `1235` checkpoints are newer than latest rating; queue length is
  `622`. Active `round-000034` still reports `4414 / 28308` root completed
  games and the cheap liveness sample is stale. A patched `drain-if-ready`
  recovery scan is running; do not infer failure from the website alone.
- 11:26 recovery result: `round-000034` published (`2739` checkpoints,
  `28308 / 28308` games), and next rating call
  `fc-01KRV8JHYV02SJ8GNH1SJPGQW4` was spawned for `3427` checkpoints. Trainer
  export is still stale from `round-000033`; `refresh-if-ready` is running.

2026-05-17 06:44 EDT:

- `round-000031` skipped under old/stale in-flight code; its skip record still
  had `skip_scan_output_progress=false`.
- A current-code drain request produced child call
  `fc-01KRTR2B7BPV74FCAY1DPMD68M`, but it stayed pending.
- Modal app logs showed the deployed tournament app was unhealthy: repeated
  runner disappearance / preemption and thousands of outstanding tasks.
- Stopped the deployed tournament app to clear that live scheduling state.
- Redeployed current tournament code successfully. Next gate: use
  `curvytron_live_loop_control.py` for status/drain. Do not claim loop
  validation until rating advances beyond `919`, trainer-facing export advances
  from that rating, and trainer-proof shows trainers loading that export.
- Fresh status after redeploy says `ready_for_next_rating_batch`: no active
  game batch, queue length `0`, intake `2121`, latest rating `919`, so `1202`
  checkpoints need rating. Running one fresh `drain-if-ready` is the correct
  next step.
- Fresh drain spawned `fc-01KRTRQJBC6TTPYS1DYZ6RVMBF`, but no active game batch
  appeared. Logs exposed the real blocker: adaptive pair selection crashed with
  `OverflowError: math range error` in a naive sigmoid on the large roster.
  Stable sigmoid fix is local and focused adaptive tests are green (`18
  passed`). Next: stop the crashing deployed app state, redeploy, then rerun
  status/drain.
- Stopped crashing app `ap-IrdpT1NJRAr9T3O26dkJFt` and redeployed the fixed
  scheduler. Next: status, then exactly one fresh drain if ready.
- Used `--ignore-drain-request-lease` once because the old leased call belonged
  to stopped/crashing app state. Fixed-code drain spawned
  `fc-01KRTRZ6EYAMPTMX79753ZNN8M`.
- Status now shows active internal game-batch artifact `round-000032` with
  `2148` checkpoints, `1053` pairs, and `22113` games. It is current-pool
  coverage, not all-pairs. Next proof: game outputs, rating publish beyond
  `919`, trainer export refresh, trainer-proof.
- Two-minute recheck: game outputs are landing. Probe saw `21` completed game
  summaries, logs show `ok=true` games with balanced randomized seats and
  `max_steps=1048576`. Latest rating is still `919`; wait for reduce/publish.
- Five-minute recheck: old recovery still skipped `round-000032` as
  `different_spec_zero_output` even though logs proved games were running.
  Cause is stale Modal Volume view during recovery output scan. Patched
  `_rating_round_skip_decision` to reload the tournament Volume before scanning
  real outputs; focused slice `23 passed`. Next: stop bad app state, redeploy,
  drain again.
- Stopped `ap-aAZ5lOtiAfWmsf828BSY0Y` and redeployed the Volume-reload fix.
  Next: compact status, then one fresh drain if ready.

2026-05-17 06:30 EDT:

- After a 5-minute sleep, `round-000031` is still running with fresh output;
  it has not been skipped by the new recovery path.
- Recent skipped batches `round-000024` through `round-000030` all show
  `skip_scan_output_progress=false`, which points to old/stale in-flight code.
- Latest rating is still `919`; no trainer refresh should be run yet.
- Next gate: continue waiting/rechecking until `round-000031` publishes,
  reduces, or gives a new skip reason from the deployed code.

2026-05-17 06:22 EDT:

- Compact recent history showed `round-000024` through `round-000030` were
  skipped as `zero_progress_smaller_pool`; latest rating stayed `919`.
- Patched recovery scan to force per-game summary counting when deciding
  whether a stale-looking batch is skippable.
- Added regression coverage for fresh per-game output preventing a zero-progress
  skip. Focused slice: `9 passed`.
- Redeployed the tournament app. Next gate: recheck live and see whether
  `round-000031` or the next batch now progresses/reduces instead of being
  skipped as zero progress.

2026-05-17 06:12 EDT:

- Redeployed the tournament app after adding active-batch spec/roster counts.
- Focused test slice is now `9 passed`.
- Live status says `round-000030` has `1095` checkpoint inputs,
  `1095` embedded rating-spec checkpoints, and `1095` roster rows.
- Latest rating is still `919`; intake is `1925`.
- Next gate remains: wait/recheck until `round-000030` publishes beyond `919`
  or safely recovers.

2026-05-17 06:10 EDT:

- Added local/server `pool_status` tooling and focused test coverage.
- Live recheck moved from suspicious `round-000029` to active `round-000030`.
- `round-000030` reports `1095` checkpoint inputs, which is larger than latest
  rating `919`, and has fresh probed game output. This means the active batch
  now includes some newer checkpoints.
- Still not done: intake is `1907`, latest rating is `919`, and no rating
  beyond `919` has published yet. Next proof is rating publish, trainer refresh,
  and trainer-proof from the newer rating.

2026-05-17 06:05 EDT:

- Latest live concern: `round-000029` is active and producing game summaries,
  but it reports only `919` checkpoint inputs.
- Latest published rating is also `919`, while intake is about `1888`.
- This means we have not proven that new checkpoints are entering the active
  rating games. The next code/tooling change is to make this mismatch explicit
  in `loop-status` / `curvytron_live_loop_control.py` instead of discovering it
  manually.
- Do not call the feedback loop validated until a later active batch clearly
  includes the newer checkpoint pool, publishes a rating beyond `919`, and
  trainer-proof shows trainers loading that newer export.

2026-05-17 05:56 EDT:

- Fixed local duplicate-drain guard after synchronous drain returns.
- Verified repeated `drain-if-ready` now returns
  `blocked_recent_drain_request` while pending rating call
  `fc-01KRTNPSZY7MCMPFMQJ9N1EAQM` is still inside the lease window.
- Next check: wait for the rating call to create a new active bounded game
  batch, or let the lease expire before considering another drain.

2026-05-17 05:54 EDT:

- Stale recovery cleared `round-000028`; status became
  `ready_for_next_rating_batch`.
- Ran exactly one deployed `drain-if-ready`.
- Drain spawned rating call `fc-01KRTNPSZY7MCMPFMQJ9N1EAQM` for `1862`
  checkpoints, `943` newer than latest rating.
- Next check: confirm a new active bounded game batch appears and either writes
  game summaries or recovers cleanly.

2026-05-17 05:49 EDT:

- Trainer-proof for generation `5` finished.
- `25/136` runs currently have a generation-5 target assignment SHA as latest
  applied.
- Provider rows for generation-5 targets: `53473` total, `23183` ok, `0` false.
- Same two runs still have `kept_previous` with assignment JSONDecodeError:
  `cz26a-r001...` and `cz26a-r020...`.
- Tooling cleanup done locally: `trainer-proof` row output is compact by
  default. Use `--assignment-proof-row-limit -1` for full rows.

2026-05-17 05:44 EDT:

- Latest rating remains `round-000015` / `919`.
- Trainer refresh advanced to generation `5`, but still from
  `round-000015`; this is not new rating progress.
- Current internal game-batch artifact is now `round-000028` with `1361`
  checkpoint inputs, `659` pairs, and `13839` games. No output seen yet by the
  bounded probe.
- Next: trainer-proof generation `5` while waiting for `round-000028` output or
  safe recovery.

2026-05-17 05:40 EDT:

- Status recheck still says `rating_game_batch_active`; no duplicate drain.
- Latest rating remains `round-000015` / `919`.
- Intake is `1766`; `847` checkpoints are newer than latest rating.
- Current internal game-batch artifact `round-000027` has real output:
  bounded probe found `6` completed game summaries even though root progress
  still says `0` completed.
- Current next action remains wait/recheck. The deployed tool must decide
  recovery; do not skip from stale root progress by hand.

2026-05-17 05:35 EDT:

- The stale-default loop was fixed and deployed. The current truth is no longer
  "export maybe exists"; generation `4` definitely exists and was written from
  rating `round-000015` / `919`.
- Trainer-proof is now real tooling, not a manual guess:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action trainer-proof --activity-probe-pairs 0 --run-limit 0`.
- That proof scanned all `136` runs. `48` currently have one of the generation
  `4` target assignment SHAs as their latest applied assignment. There are
  `43723` target provider-ok env rows and `0` target provider-false rows.
- This proves tournament output has started coming back into trainers. It does
  not prove every trainer has refreshed yet.
- The rating side is still behind intake: latest rating `919`, intake `1763`.
  Current internal game-batch artifact `round-000027` has real output. Do not
  spawn duplicate work; wait/recheck until it publishes or the deployed status
  says recovery is safe.
- Keep the next status/proof facts in this doc set before doing more control
  actions.

2026-05-17 05:21 EDT:

- CZ26 defaults are now patched and deployed. The deployed refresh path no
  longer falls back to old r18fresh config.
- `refresh-if-ready` succeeded after deploy:
  - generation `4`;
  - source `round-000015`;
  - source rows `919`;
  - active rows `100`;
  - rewritten pointers `24`;
  - snapshot `auto-r000015-g4-ec1bef62`.
- The loop is now proven through trainer-facing export once, but not fully
  caught up. Intake has `1660` checkpoints and latest rating has `919`, so
  `741` checkpoints remain newer than the latest rating.
- Active game-batch artifact `round-000026` has `1307` checkpoint inputs,
  `632` pairs, and `13272` games, but status currently sees zero completed
  game summaries. Next action is to watch this batch for output/recovery, then
  advance ratings beyond `919` and refresh trainer pointers again.

2026-05-17 05:14 EDT:

- Rating did move forward after the far-ahead blocker patch. Latest is now
  `round-000015` / `919` checkpoints, up from `round-000010` / `588`.
- The full loop is still not validated because trainer refresh is stale:
  generation `3` still points at `round-000007`.
- The immediate blocker is stale shared defaults. The deployed app still uses
  r18fresh as the current-lane fallback for trainer-candidate refresh, so
  `refresh-if-ready` read the wrong config and failed with
  `training candidate refresh config has wrong schema_id: None`.
- Correct CZ26 refresh config is:
  `control:training/lightzero-curvytron-visual-survival/cz26-control/attempts/try-cz26-control/opponents/training_candidate_refresh_config.json`.
- Correct CZ26 refresh pointer count is `24`, from the
  `cz26-full-20260517a` manifest.
- Next action: patch `src/curvyzero/contracts/curvytron.py` and locked tests to
  make CZ26 the actual current default, redeploy, then run `refresh-if-ready`
  and prove trainer refresh advances beyond generation `3`.

2026-05-17 05:01 EDT:

- The live loop is blocked by a hidden current-lane artifact, not by old
  tournaments. Logs show repeated rating-loop no-ops on `round-000024` with
  `existing_input_different_spec` (`pool_hash`/`roster_hash` mismatch).
- The status tool missed it because it scanned only from latest
  `round-000010` through `round-000022`. The blocker is farther ahead.
- OBS-2 next action is now concrete: status must discover all unrated blocking
  game-batch artifacts, and drain recovery must skip zero-output
  different-spec artifacts so the current checkpoint pool can create useful
  games.
- Local code now implements that status/recovery patch and focused tests pass.
  Next action is deploy, then run deployed status/drain through
  `scripts/curvytron_live_loop_control.py`.
- Patch deployed. Safe drain spawned rating loop
  `fc-01KRTK0AMKAM34BW3FKYBAYR0C` for `1626` checkpoints. The next check is
  whether a new current-code game batch appears and starts writing outputs.
- Watch out for misleading old output: logs show old workers from skipped
  `round-000022` still writing games. That is cleanup/noise unless it prevents
  the new batch from scheduling.
- Do not claim the loop is healthy until latest rating moves beyond
  `round-000010` / `588` checkpoints and trainer refresh moves beyond
  generation `3`.

2026-05-17 04:29 EDT:

- Still not validated. Safe deployed-app status at `08:29Z` shows active
  `round-000022` with `1361` checkpoint inputs, `659` pairs, `13839` games, and
  `0` completed game summaries visible to the probe.
- Intake is alive but backed up: `1381` checkpoints known, `793` not in latest
  rating, queue length `85`.
- Rating/export are stale: latest rating remains `round-000010` / `588`
  checkpoints; trainer refresh remains generation `3` from `round-000007`.
- Logs show timeout cancellations. Old `1800s` cancellations are expected from
  pre-patch workers, but newer `600s` cancellations are not yet explained. Next
  task is to identify the function path causing `600s`, then use the safe
  deployed-app control script to let recovery advance without duplicate spawns.
- Current operator command:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 12`.
- `08:32Z`: safe `drain-if-ready` spawned one bounded drain,
  `fc-01KRTH2CJ483F5RYV945MZ4BQH`, after seeing no active game batch. Do not
  spawn another one unless the control script says it is ready again. Next proof:
  new active batch plus visible game summaries.
- Tooling cleanup: the safe control script now records a `600s` operator
  drain-request lease in the intake Dict before spawning, so repeated operator
  calls block instead of creating duplicate drain requests while Modal is still
  scheduling.
- `08:39Z`: no active batch appeared after the earlier drain request; queue grew
  to `482`. Re-ran guarded `drain-if-ready`, spawned
  `fc-01KRTHECNW3S4EMBPZ5A3QX6JE`, and recorded the lease. Next status check
  should show either an active batch or the lease blocking duplicate spawn.
- `08:42Z`: queue was `0`, but no active game batch existed. The detached drain
  consumed events but did not visibly spawn the rating loop.
- `08:43Z`: synchronous deployed drain result repaired a stale rating claim and
  spawned rating loop `fc-01KRTHMV0DN5TH8SBZM6EM945Y` for `1494` checkpoints.
  Next proof: active game batch and visible game summaries.
- `08:44Z`: still no active game batch; latest remains `round-000010` / `588`.
  Function-call log lookup for `fc-01KRTHMV0DN5TH8SBZM6EM945Y` returned no
  lines, so it may not have scheduled yet.
- `08:49Z`: post-deploy status still shows no active game batch. Queue is `12`;
  latest remains `round-000010` / `588`. Wait for the current drain-request
  lease to expire around `08:53Z`, then re-run synchronous `drain-if-ready` if no
  active batch appears.

2026-05-17 04:01 EDT:

- Still not validated. The live loop needs one current-code bounded game batch
  to complete and publish a newer snapshot before we can trust it.
- Normal live operations should now use
  `scripts/curvytron_live_loop_control.py`, not `modal run`, because
  `modal run` creates a temporary scheduled app and that temporary app caused a
  real `ConflictError` while trying to spawn rating work.
- Focused local validation for the status/control path passed: `43 passed`.
- Deployed-app script status at `08:01Z`: one active bounded game batch,
  `round-000020`, `1228` checkpoint inputs, `593` pairs, `12453` games,
  `save_gif=false`, real game output found (`21` summaries in the activity
  probe), no duplicate active batch. Latest rating remains `round-000010` /
  `588`; trainer refresh remains generation `3`.
- Next check:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 12`.
- Do not spawn another drain while `round-000020` is active. Wait for completion
  or recovery, then prove latest rating and trainer refresh advance.

2026-05-17 03:50 EDT:

- Earlier plan was to use `modal run --mode loop-control`; that is now
  superseded by the deployed-function script above.

2026-05-17 03:10 EDT:

- Do not mark the loop validated. `round-000015` contradicted the intended
  safety rule: `loop-status` saw real output at `07:04Z`, but by `07:07Z`
  the batch was marked skipped and latest rating was still `414`.
- Current likely explanations: an old in-flight drain used stale code, or the
  full recovery scan still missed live output. We need the next batch to prove
  which one is true.
- Immediate code/tooling task is locally done: each skipped game-batch
  `skip_decision` is exposed in `loop-status`. It shows completed game count,
  latest result timestamp, stale age, whether output scan was used, and scan
  errors. Focused local validation passed: `39 passed`.
- Historical 03:10 note said to run `intake-drain --detach`; current safe path
  supersedes that. Use `scripts/curvytron_live_loop_control.py --action
  drain-if-ready` for live drain control, then verify the next bounded game
  batch writes games, reduces ratings, updates trainer export, and is loaded
  by trainers.
- Fast status at `07:21Z`: latest rating is now `round-000011` with `681`
  checkpoints; active game batch is `round-000017` with `960` checkpoint
  inputs; intake has `1063`; queue has `83`; trainer refresh is still
  generation `3` from `round-000007`. Old skipped batches report
  `scan_output_progress=false`, so those were skipped by stale code.
- `07:22Z` activity probe confirmed `round-000017` has live output. `07:26Z`
  status showed it had not reduced yet and also exposed `round-000016` as a
  second un-rated active artifact. `loop-status` now reports active-batch
  count and flags multiple active game batches. Focused tests: `40 passed`.
- `07:30Z` status showed `round-000017` was skipped by stale in-flight code
  (`scan_output_progress=false`) and latest had regressed to `round-000010` /
  `588` checkpoints. Patched parent `rating_loop` to stop bypassing the
  monotonic latest guard. Focused tests: `44 passed`.
- `07:36Z` status after detached drain: active `round-000018`, `1095`
  checkpoint inputs, real output seen, latest still `round-000010` until this
  or another current-code reduction publishes.
- `07:43Z` status after wait: `round-000018` was skipped by stale code
  (`scan_output_progress=false`). The scheduled drain tick was running full
  drain logic through `.local()` and then trying to spawn work from a stopped
  app. Patched scheduled drain tick to spawn the drain function instead of
  mutating state inline. Focused tests: `45 passed`.

2026-05-17 02:41 EDT:

- Corrected the recovery model. A zero-started compact progress file can be
  false while Modal game workers are still writing outputs. Logs showed this
  happened for `r000012`.
- Current code patch wires `scan_output_progress=True` into the actual
  intake-drain recovery call. If the output scan fails, the recovery decision
  must not silently skip from stale control files.
- Focused local validation for this patch passed: `36 passed`.
- `round-000013` spawned with `815` checkpoint inputs after the continuation
  pool fix. Next proof is whether it starts/completes/recovers without
  dropping active game output, then produces a rating snapshot and trainer
  refresh.
- Live `loop-status` at `06:47Z` showed intake `828`, latest rating `414`,
  `round-000013` skipped, and active `round-000014` with `681` checkpoint
  inputs. Logs showed `r000014` games actively completing, so the next cleanup
  is to make `loop-status` scan and report real game output directly.
- Live `loop-status` at `07:04Z` now reports active output directly: intake
  `956`, latest rating `414`, active `round-000015` with `919` checkpoint
  inputs, and `21` game summaries found by the bounded activity probe. Next
  gate is rating completion/reduction and trainer refresh beyond generation
  `3`.

2026-05-17 02:02 EDT:

- `--mode loop-status` is deployed and locally tested. Focused suite:
  `34 passed`.
- The recovery timers are now separated. Intake claim ownership remains
  conservative at `24h`; apparently stuck tournament game-batch recovery uses
  `600s` plus a real game-output scan before skip.
- Live `loop-status` at `05:58Z`: intake `549`, latest rating `414`,
  queue `18`, active internal game batch `round-000008`, zero started.
- After the timer split deploy and a detached drain, `round-000008` completed
  and ratings were written. Live `loop-status` at `06:02Z` showed latest rating
  at `round-000008`, `414` checkpoints.
- Historical at that read: active internal game batch was `round-000009`.
  It later completed, so do not use this as the current target.
- Intake is now `564`, latest rating still `414`, queue `7`; therefore `150`
  newer checkpoints are not yet represented in latest ratings.
- Follow-up `loop-status` at `06:11Z` proved the recovery path worked:
  `round-000009` completed, `round-000010` was skipped after staying
  zero-started, and `round-000011` spawned with `681` checkpoint inputs. Queue
  length was `0`.
- Follow-up `loop-status` at `06:26Z` exposed a continuation-pool regression:
  after `round-000011` skipped, `round-000012` spawned with only `414`
  checkpoints while intake had `705`. Patched continuation pool selection to
  use current manifest `checkpoint_refs` plus any `seen_checkpoint_refs`, not
  stale `seen_checkpoint_refs` alone. Focused tests stayed green: `34 passed`.
- Next check now: after `round-000013` starts/completes/recovers, verify the
  current intake pool is still used and real game output is not skipped.

2026-05-17 01:55 EDT:

- Do not keep using raw Modal Volume reads as the normal status path. The
  current fix is a compact operator tool: `--mode loop-status`.
- Historical before deploy: `loop-status` was local code then and needed
  testing/deploy. It is now deployed and reports the actual live manifest,
  intake count, queue length, latest rating summary, current internal
  game-batch artifact, trainer refresh state, and recommended next action.
- Duplicate tournament calls with an existing active game-batch input now back
  off with `running_existing_round` / `existing_input_different_spec` instead
  of crashing or overwriting.
- Bounded tournament game batch `round-000006` completed: `300` pairs,
  `6,300` games, `0` failed games, ratings written.
- Trainer-candidate auto-refresh succeeded after that path: source
  `round-000007`, `414` rows, `100` active rows, generation `3`, snapshot
  `auto-r000007-g3-b2b6f2bc`, and `24` refresh pointers rewritten.
- Intake later reached `545` checkpoints. Latest trainer-refresh source had
  `414`, so about `131` newer checkpoints still need tournament rating coverage
  before they can affect trainer opponents.
- Historical possible stall: `round-000008` later completed, so do not treat it
  as current.

2026-05-17 01:36 EDT:

- Stop saying "round" as if it is a training concept. In tournament files,
  `round-000006` is just an internal game-batch directory. Trainers are
  continuous and keep writing checkpoints.
- The deployed tournament app now repairs bad live manifest state on dict load,
  volume load, seed, submit, and subscriber tick, and live all-pairs is capped
  even at exactly `100` checkpoints. Targeted repair suite: `32 passed`.
- Real live state now reads: active manifest only
  `manifest:cz26-live-20260517a:elo-cz26-live-20260517a`, `checkpoint_count=414`,
  `seen_checkpoint_count=414`, `queued_checkpoint_count=414`, queue length `0`,
  `pair_selection=adaptive_v0`, `pairs_per_round=300`,
  `active_pool_limit=100`.
- After the final hardening deploy, status read `checkpoint_count=431`, queue
  length `17`, still `adaptive_v0 / 300 / 100`.
- Subscriber tick after deploy found two more checkpoints and committed
  successfully. This proves the subscriber is still seeing fresh checkpoints.
- Drain skipped stale zero-progress bounded game batch `round-000005` and
  spawned rating call `fc-01KRT6GW1QABPAHZSG4TXN8RNY`.
- Current tournament game batch is `round-000006`: `300` pairs, `6,300` games.
  Modal logs show active `r000006` games completing with `ok=true`, balanced
  random seat order, and `max_steps=1048576`.
- Do not run another manual drain with `claim_stale_after_seconds=60` against
  the live running batch. That knob was for old dead batches; the deployed
  default stale window is `24h`.
- The root progress JSON lags during the game map and still reported zero
  completed games at `05:30:17Z`; logs prove workers are running. Completion
  proof still requires ratings/latest output after reduce.
- Next gate: wait for `round-000006` to finish, verify `latest.json` advances
  past the old `340`-checkpoint snapshot, publish/refresh the trainer
  leaderboard, and verify trainers consume the new assignment generation.

2026-05-17 01:20 EDT:

- The full `cz26-full-20260517a` batch is still the active line. Trainers are
  alive and producing checkpoints; an independent read-only sanity check found
  all `136` manifest rows with heartbeat/status evidence, running pollers, and
  checkpoints present.
- The active intake control plane was purged. The real `shankha-dev` Modal dict
  now has exactly one active manifest key:
  `manifest:cz26-live-20260517a:elo-cz26-live-20260517a`.
- Seven stale intake manifest records were removed from the Modal dict. The
  live ephemeral apps from stale attempts were stopped; deployed tournament,
  trainer, and browser/control services remain.
- Root cause of the latest stall: the current `cz26` manifest itself was still
  configured as one giant live all-pairs batch: `pair_selection=all_pairs`,
  `pairs_per_round=4950`, `games_per_pair=21`, about `103,950` games per
  update. Deploying fixed defaults did not rewrite the existing Modal dict or
  volume manifest.
- Corrected live design: live intake uses bounded continuous scheduling:
  `pair_selection=adaptive_v0`, `pairs_per_round=300`,
  `active_pool_limit=100`, `games_per_pair=21`, `games_per_shard=1`. The
  tournament can still improve rankings over time, but one giant batch cannot
  freeze the feedback loop.
- Both live manifest copies were repaired. A later read exposed another gap:
  valid-looking bad Modal Dict state was returned without repair. That path is
  now patched too.
- The stuck giant tournament game batch `round-000003` was marked `skipped`
  with reason `stale_incomplete_smaller_pool` after only `4 / 103,950` games.
  This is intentional cleanup, not a valid rating result. `round-000003` is
  not a training round.
- New bounded tournament game batch `round-000004` was spawned with `300` pairs
  and `6,300` games under call `fc-01KRT5WRCAAPSACZZYNZETR7W3`.
- Trainer consumption is partially proven beyond seed state: a sampled trainer
  applied assignment `cz26-auto-recipe-07`, whose assignment source resolves to
  tournament refresh generation `2`.
- Open check now: wait for the bounded tournament game batch to complete,
  confirm `latest.json` advances, publish/refresh the trainer leaderboard, and
  verify trainers consume the next refreshed generation.

2026-05-17 00:31 EDT:

- The full current-code batch is launched, not just planned:
  `cz26-full-20260517a`, `136` rows total (`96` Grid A + `40` Grid B).
- Launch manifest:
  `artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.json`.
- Submit output:
  `artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.submit_launch.json`.
  It contains `136` `spawned` records with train and poller function-call ids.
- Shared initial policy seed for every row is the old overnight rank-1
  checkpoint preserved in `TOP10_RAW_REFS_auto-r000032-g22-555c999b.txt`.
- Tournament lane is current:
  - tournament `cz26-live-20260517a`
  - rating `elo-cz26-live-20260517a`
  - trainer/public leaderboard
    `cz26-live-20260517a-elo-cz26-live-20260517a-training`
- Seed tournament round completed: `4` seed checkpoints, `6` pairs, `126`
  games, `0` failed games, ratings written.
- Seed leaderboard published to the public Modal dict with `4` active rows.
- Training-candidate auto-refresh ran against that seed leaderboard and rewrote
  all `24` current refresh pointers. This proves the seed leaderboard can flow
  back into trainer opponent assignments.
- Historical mistake: intake manifest was briefly forced to
  `pair_selection=all_pairs`, `pairs_per_round=4950`,
  `active_pool_limit=100`, `games_per_pair=21`, and `games_per_shard=1`.
  Current live scheduling is bounded `adaptive_v0 / 300 / 100`.
- All `136` top-level `cz26a`/`cz26b` run directories are present in the v2
  runs volume.
- Open check now: count how many of the `136` spawned train calls have created
  their attempt/train contents, progress files, and first checkpoints. Early
  sample rows are running; some late sample rows had only top-level directories
  at the first check, which may be Modal startup/backlog or early failure.
- Next gate: finish the all-run status sweep, inspect logs for failed train
  calls, and re-submit only genuinely missing/dead rows if needed.

2026-05-17 00:43 EDT update:

- Lightweight all-run volume check is green for launch reality:
  `136/136` run trees, `136/136` attempt dirs, `136/136` train dirs, and
  `136/136` rows with at least one checkpoint. `133/136` had
  `progress_latest.json`; the three missing progress sidecars still had
  checkpoints.
- Subscriber automation is live. Intake advanced from seed-only to current
  `checkpoint_count=140` and queued/drained new `cz26` checkpoints without a
  manual checkpoint list.
- Cleaned active intake rotation in the real `shankha-dev` Modal dict so only
  `manifest:cz26-live-20260517a:elo-cz26-live-20260517a` remains active.
  Earlier cleanup attempted the wrong environment first; that did not affect
  the deployed scheduler. The corrected write used explicit
  `environment_name="shankha-dev"`.
- Deployed drain recovered a stale claim and spawned rating continuation
  `fc-01KRT3W7QNKNH4EVZ6N66WFR9S`.
- Historical tournament game-batch state at 00:43 EDT:
  - `round-000000`: seed game batch complete, `6` pairs, `126` games.
  - `round-000001`: seed continuation game batch complete, `6` pairs,
    `126` games.
  - `round-000002`: completed, `903` pairs, `18,963` games, ratings written.
  - `round-000003`: then running, `4,950` pairs, `103,950` games.
- This 00:43 next gate was superseded by the 01:20 repair. Do not wait for
  `round-000003`; it was skipped as the stale giant tournament game batch.

2026-05-17 00:11 EDT:

- Historical pre-launch note: at this point the full batch was still gated on
  deployed-loop canary evidence. That is no longer the current state; the batch
  above has now been launched.
- Fixed checkpoint intake merge behavior: a reseed with explicit old seed refs
  plus a live run watch now upgrades the durable manifest instead of staying as
  a static seed-only manifest.
- Fixed `training-candidate-auto-refresh` CLI wiring so canaries can pass an
  explicit `training_candidate_refresh_config_ref` and avoid accidentally
  reading the broad current-lane config.
- Fixed rating-writer liveness: a completed `latest.json` snapshot now wins over
  a stale root `progress.json` for the same or older round. This was blocking
  canary continuation because `latest.json` had completed `round-000009`, while
  `progress.json` still said that same round was running.
- Current canary state:
  - tournament `cz26c-e2e-20260516a`
  - rating `elo-cz26c-e2e-20260516a`
  - trainer run `cz26c-r001-out100-n0-imm0-b20w05r1`
  - intake manifest is live-watch now: `run_id_prefix=cz26c-`,
    `selection=explicit_refs_plus_run_watch`, `checkpoint_count=6`.
  - queue drained to zero and rating continuation spawned
    `fc-01KRT21JW6AFJ0T8E883VMTN5W`.
  - continuation tournament game-batch input `round-000010` has
    `6` checkpoints, `15` pairs, and `315` games; it includes the new `cz26c`
    checkpoints.
- Local validation after these fixes is green: focused 15-test patch suite
  passed, then broader loop/tournament/provider suite passed
  `338 passed, 14 skipped`.
- Next gate: wait for canary tournament game batch `round-000010` to finish,
  publish the canary training leaderboard, run canary training-candidate
  refresh with its explicit config, and prove the trainer consumed the new
  assignment SHA.

2026-05-16 21:45 EDT:

- Cleaned the next-batch contract instead of adding another manual loop. New
  launch manifests publish three things as one bundle: opponent assignments,
  refresh pointers, and the training-candidate scheduler config.
- Added `out50` as a first-class reward tag because Grid B is fixed at
  `reward_outcome_alpha=0.5`.
- `scripts/build_curvytron_next_batch_manifest.py` now emits the current
  `cz26` family: `cz26c` canary, 96-row `cz26a` Grid A, and 40-row `cz26b`
  Grid B. Grid B row names start at `r001`, even when emitted in the full
  136-row manifest, so operator-facing names are local and readable.
- Every row seeds from the rank-1 checkpoint in the source leaderboard snapshot.
  Leaderboard immortality is a separate slot flag; the policy remains rank1,
  rank2, etc.
- Focused contract validation is green: `11 passed` for naming, next-batch
  manifest shape, and submitter dry-run publishing of scheduler config.
- Broader focused local validation is green: `448 passed, 24 skipped`. This
  includes shared current constants, training-candidate refresh, lineage chain,
  intake, rating, publish, trainer refresh plumbing, opponent mixture/leaderboard
  handling, source-state env behavior, and GIF browser defaults.
- The broader run initially caught a real miss: `_run_visual_survival_train`
  required `reward_outcome_alpha` at the internal call boundary. Fixed it to use
  the shared default and reran the suite green.
- Next gate: deploy the updated tournament and trainer apps, then run the
  automated canary end to end before launching the full batch.

2026-05-16 18:42 EDT:

- Local current-lane defaults briefly drifted to `slot64-*` refresh pointer
  names while the live r18fresh control volume still had the actual
  `blank10-*` / `blank20-*` pointers. Fixed the shared current contract back to
  the live pointer paths and added a shared-contract regression guard.
- The live r18fresh current tournament is not intake-stuck: intake status shows
  `585` checkpoint refs and queue length `0`.
- Rating latest is `round-000062`, `585` rows, `100` active rows, max checkpoint
  iteration `310893`.
- Trainer-facing training-candidate refresh was validated with the scheduled
  tick function: generation `38`, snapshot `auto-r000062-g38-3f471334`, three
  pointer rewrites, active count `100`.
- Fixed a reducer recovery bug found by tests: game-summary-file progress no
  longer forces the shard-tally reduce path when shard summaries are absent.
- Focused validation is green: `272 passed, 11 skipped`. Next gate is deploying
  the cleaned tournament app, then running a deployed current-code canary whose
  pointer is the same one consumed by the trainer.

2026-05-16 18:09 EDT:

- All active ephemeral `curvyzero-checkpoint-tournament-v2` apps from stale
  detached runs were stopped.
- Deployed services were kept running: tournament, trainer, and GIF browser.
- Volume cleanup was deliberately narrow: only transient `curvy-scale-probe-*`
  tournament directories were deleted.
- Preserved r18fresh/ bounded r18fresh tournaments, r18fresh training runs,
  e2e proof artifacts, and control pointers.
- The deployed canary still has a real loop gap: rating ran, but the refresh
  rewrote the wrong control pointer and the trainer/public leaderboard file was
  stale.

2026-05-16 16:23 EDT:

- Focused local proof reran green:
  `8 passed` for the synthetic loop, random learner-seat regression, and shared
  CurvyTron contract tests.
- Current Modal objects exist as all-v2 assets: runs, tournament, control
  volumes; checkpoint-intake queue; checkpoint-intake and opponent-leaderboard
  dicts.
- Deployed apps are present for tournament, trainer, and GIF browser, plus one
  detached tournament app from the r18fresh live run.
- This does not replace the P0 deployed canary. The current launch gate remains:
  prove one fresh current-code checkpoint through deployed Modal Volume/Dict
  state from write -> intake -> rating -> leaderboard/export -> trainer
  provider load with the same assignment SHA.

## Delegation Rules

- Main thread owns integration, docs, destructive decisions, and final status.
- Workers/explorers get bounded questions and a named integration target.
- Do not spawn a new lane if a current lane answers the same question.
- Close completed lanes quickly; saturated agent pools are a real risk.
- Every long wait needs a sleep ticket in `ORCHESTRATION.md` or the experiment
  log.

## Current North Star

The next large run should not start because we are impatient. It should start
because a canary or synthetic proof shows the whole loop is observable and the
next-batch design says exactly what signal we expect to learn.
