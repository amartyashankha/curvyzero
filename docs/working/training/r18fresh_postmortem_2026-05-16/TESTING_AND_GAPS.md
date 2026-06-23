# Testing And Gaps

This is the current validation map for tournament/trainer loop cleanup.
For the current cz26 runtime truth, see `CZ26_CURRENT_TRUTH_2026-05-17.md`.

## Existing Coverage

- Checkpoint production/progress: mocked LightZero save hooks, progress files,
  checkpoint metadata, commit-on-checkpoint, live publisher spawn ordering, and
  own-checkpoint pointer writing.
- Intake/subscriber/drain: live watch defaults, bounded adaptive ratings, queue
  repair, continuation from latest, and retired-history preservation.
- Tournament scheduling/rankings: unordered all-pairs, randomized seats,
  bounded adaptive scheduling, batch Elo, placement status, pair history,
  scheduler state, and leaderboard publish.
- Leaderboard/materialization: active/provisional/retired rows, stable slots,
  immortal blank/wall sentinels, context checks, pointer rewrite, and
  trainer-visible refresh pointers.
- Trainer opponent refresh: due buckets, reset params, all-env ready proof,
  retry/failure behavior, hook install, and env-level mixture refresh.
- Player role and observation contract: random-per-episode seat mode, fixed-seat
  diagnostics, source-state metadata, and tournament rejection of legacy/GPU
  observation surfaces.
- Immortal behavior: ego can still die, opponent wall/body death can be blocked,
  and public slot recipes use `opponent_immortal` rather than leaking lower
  level env switches.
- GIF/browser defaults: picker flags, current/archive default, newest
  successful GIF selection, background GIF config, and full-size tournament GIF
  policy.

## Fast Local Sweep

Latest live validation evidence: 2026-05-17 09:08 EDT.

- Honest current state: one complete live feedback pass has been proven; the
  next pass is active but not yet published.
- Proven pass:
  1. `round-000033` published as latest rating.
  2. `refresh-if-ready` wrote trainer-facing generation `11` from
     `round-000033`.
  3. `trainer-proof` found generation-11 assignments in real trainer provider
     rows.
- Active next pass:
  - `round-000034` is running;
  - active batch covers `2739` checkpoints, `1348` pairs, and `28308` games;
  - liveness probe found fresh game output;
  - latest rating remains `round-000033` / `2192` until reduction/publish.
- Open validation gap: prove `round-000034` publishes beyond `2192`, refresh the
  trainer-facing export from it, then rerun `trainer-proof`.

Trainer-side health evidence: 2026-05-17 09:25 EDT.

- Added a fast run-status path because the old all-run status table timed out at
  `300s` when asked to scan all `136` rows.
- Fast summary result:
  - row count: `136`;
  - heartbeats: `136`;
  - progress_latest files: `136`;
  - running: `134`;
  - failed: `2`;
  - failed run ids:
    `cz26a-r013-out67-n0-imm0-b20w05r1`,
    `cz26b-r028-out50-n10-imm10-b20w05r1`;
  - failed-row detail:
    - `cz26a-r013...`: `36` checkpoints, latest `iteration_170000`;
    - `cz26b-r028...`: `8` checkpoints, latest `iteration_30000`;
    - fast heartbeat/status fields do not expose a failure reason;
  - checkpoint count range: `8..36`;
  - total checkpoint artifacts: `2910`;
  - latest progress iteration range: `30000..220000`.
- Interpretation: the main Grid A/Grid B batch is mostly alive and producing
  checkpoints. Two failed rows need later diagnosis, but they do not explain the
  tournament lag by themselves.
- Grid B contains pure slot controls `b100`, `w100`, and `r1` inside the main
  `136` rows. The separate one-row `cz26c` canary completed with `350`
  checkpoints and latest progress iteration `17385`.

Tournament-side evidence: 2026-05-17 09:25 EDT.

- Active game-batch artifact `round-000034` is still running:
  `2739` checkpoints, `1348` pairs, `28308` games.
- Latest published rating is still `round-000033` / `2192`.
- Intake is now `2877`, so new checkpoints are continuing to arrive while
  `round-000034` plays.
- Liveness probe saw fresh game output. This is not total completion proof.

Trainer-consumption evidence: 2026-05-17 09:47 EDT.

- Trainer-facing export is generation `12`, still sourced from
  `round-000033`.
- `trainer-proof` scanned all `136` runs:
  - latest-applied target count: `62`;
  - target assignment count: `24`;
  - target env rows: `138045`;
  - provider OK rows: `48666`;
  - provider false rows: `0`;
  - provider null/pending rows: `89379`.
- Interpretation: trainer consumption of the current tournament export is now
  strongly proven. It does not prove the next tournament publish yet, because
  `round-000034` is still running.

Trainer and tournament evidence: 2026-05-17 10:30 EDT.

- Main batch launch remains real:
  - manifest rows: `136`;
  - submit records: `136/136` spawned;
  - fast trainer status: `136` heartbeats, `136` progress files;
  - train status: `134` running, `2` failed;
  - checkpoint artifacts: `3185`;
  - latest progress iteration range: `30000..250000`.
- Trainer consumption is stronger:
  - trainer export generation: `13`;
  - source rating: still `round-000033`;
  - latest-applied target count: `102/136`;
  - target provider rows: `74170 ok`, `0 false`, `153090 null/pending`.
- Tournament status:
  - intake checkpoint count: `3193`;
  - latest published rating: `round-000033` / `2192`;
  - active game-batch artifact: `round-000034`;
  - active batch: `2739` checkpoints, `1348` pairs, `28308` games;
  - app logs show successful `round-000034` game results at 10:29 EDT;
  - a wider liveness probe saw output about `44s` old at 10:30 EDT.
- Remaining validation gap:
  - `round-000034` has not written ratings yet;
  - the expensive `--progress-probe` timed out at `300s`, so total active-batch
    completion is still not cheaply observable;
  - the next hard proof is a rating publish beyond `2192`, followed by
    `refresh-if-ready` and `trainer-proof` from that newer rating.

Recovery/control validation: 2026-05-17 11:20 EDT.

- Found and fixed two actual issues:
  - stale active batches could not enter drain recovery because active batches
    were blocked before recovery ran;
  - adaptive pair selection expanded the nominal `300` pair cap to `1348`
    pairs via the old first-touch floor.
- Deployed fixes:
  - stale active batch -> recovery scan is allowed;
  - scheduled drain tick probes activity before deciding;
  - adaptive `pairs_per_round` is now a hard cap;
  - large recovery scans use bounded activity probing;
  - recovery decision details are visible in operator output.
- Focused tests:
  - `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "adaptive_v0 or feedback_loop_control_decision or rating_round_skip_decision or active_game_batch_output_stale" -q`
    -> `19 passed`;
  - `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "feedback_loop_control_decision or active_game_batch_output_stale or drain_tick or adaptive_v0" -q`
    -> `18 passed`.
- Live recovery result:
  - `round-000034` was not skipped;
  - bounded scan found `5103 / 28308` completed games and fresh activity;
  - visible root progress now reports about `4414 / 28308` games complete.
- Remaining gap:
  - current `round-000034` is still oversized and unpublished;
  - future rounds are bounded, but this in-flight one still has to finish,
    reduce, or later become genuinely stale.

Previous live validation evidence: 2026-05-17 08:52 EDT.

- The core live feedback loop has now been observed once:
  1. `round-000033` published as latest rating.
  2. `refresh-if-ready` wrote trainer-facing generation `11` from
     `round-000033`.
  3. `trainer-proof` found generation-11 assignments in real trainer provider
     rows.
- Exact evidence:
  - latest rating: `round-000033`, `2192` checkpoints, `22575` completed games;
  - refresh snapshot: `auto-r000033-g11-50405322`;
  - trainer-facing active rows: `100`;
  - rewritten pointers: `24`;
  - trainer-proof scanned `136` runs;
  - latest-applied target count: `3`;
  - target env rows: `6663`;
  - provider OK rows: `4548`;
  - provider false rows: `0`;
  - provider null/pending rows: `2115`.
- Still open:
  - trainer catch-up is partial, not complete;
  - `522` newer checkpoints are now queued and need the next bounded rating
    batch;
  - survival/performance signal after generation-11 consumption still needs
    later measurement.
- Tooling validation:
  - focused status tests passed after the root/liveness/progress field split:
    `7 passed`;
  - first full progress scan was too slow and timed out after `300s`, so the
    default was changed back to fast status and the expensive scan is now
    explicit `--progress-probe`.

Previous live validation evidence: 2026-05-17 08:39 EDT.

- Current active internal tournament game-batch artifact is still
  `round-000033`.
- Live status facts:
  - latest rating is still `round-000015` / `919`;
  - active batch has `2192` checkpoint inputs, `1075` pairs, and `22575`
    intended games;
  - intake advanced to `2668`, so `476` checkpoints are queued for a later
    batch;
  - trainer-facing export generation is `10`, but still from `round-000015`.
- App logs show fresh `round-000033` game completions at `12:36-12:39Z`, plus
  runner disappearance/reschedule messages. This proves the active batch is
  still doing work/retries.
- Current tooling gap:
  - `probe_completed_game_count=21` is a liveness sample, not total progress;
  - `completed_game_count=0` is the unreduced root summary, not proof of zero
    per-game output;
  - the Modal call graph is capped/sample-like and cannot prove all `22575`
    game calls completed.
- Validation remains open until a rating beyond `919` publishes, a
  trainer-facing export is written from that newer rating, and `trainer-proof`
  shows running trainers loaded it.
- Next test/tooling work: expose a true progress readout or an explicitly
  labeled sampled progress readout before relying on status output for operator
  decisions.

Previous live validation evidence: 2026-05-17 07:15 EDT.

- Current active internal tournament game-batch artifact is `round-000033`.
  This is a game-batch artifact id, not a training round.
- Live status facts:
  - latest rating is still `round-000015` / `919`;
  - active batch has `2192` checkpoint inputs, `1075` pairs, and `22575`
    intended games;
  - status probe saw `21` completed game summaries, newest about `11s` old;
  - Modal call summary saw `44` tournament game calls succeeded and `52`
    pending in the sampled call graph;
  - intake advanced to `2252`, so `60` checkpoints arrived after this active
    batch started.
- Interpretation: games are actually running now. The full loop is still not
  validated until a rating beyond `919` publishes, trainer-facing refresh uses
  that newer rating, and trainer-proof shows trainers loading that export.
- Tooling cleanup: `curvytron_live_loop_control.py` now prints compact
  `proof_chain`, `blockers`, and `open_items` in status, and the default
  pending-call probe gives call-graph counts plus a small sample.

Earlier live validation evidence: 2026-05-17 06:22 EDT.

- New failure mode found by compact recent-batch history:
  smaller-than-current-desired active batches were repeatedly skipped as
  `zero_progress_smaller_pool`, keeping latest rating stuck at `919`.
- Code fix: recovery skip decisions now force per-game summary counting while
  scanning output progress. This matters because current tournament fanout uses
  one Modal worker per game, so shard summaries are not the only valid evidence
  of progress.
- Regression test added: a stale-looking smaller-pool batch with a fresh
  per-game summary is not skippable and reports completed/started counts.
- Focused test command:
  `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "skip_decision or rating_game_batch_status_summary or feedback_loop_status"`.
- Result: `9 passed`.
- Redeployed the tournament app. Remaining proof: live batch publishes beyond
  `919`, then trainer refresh and trainer-proof consume that newer rating.

Earlier live validation evidence: 2026-05-17 06:12 EDT.

- Added and deployed active-batch pool instrumentation:
  `pool_status`, `rating_spec_checkpoint_count`, and `checkpoint_roster_count`.
- Focused local validation passed:
  `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "feedback_loop_status or feedback_loop_control_decision or rating_game_batch_status_summary"`
  reported `9 passed`.
- Live status after deploy:
  - latest rating `round-000015` / `919`;
  - intake `1925`;
  - active internal game-batch artifact `round-000030`;
  - active batch checkpoint/spec/roster counts all `1095`;
  - fresh probed game output exists.
- Gap remains: no rating beyond `919` has published yet, and trainer refresh
  generation `6` still comes from `round-000015`.

Earlier live validation evidence: 2026-05-17 06:05 EDT.

- The full loop is still not validated.
- `drain-if-ready` at 05:54 reported a desired pool of `1862` checkpoints,
  with `943` checkpoints newer than the latest rating.
- The next visible active internal game-batch artifact, `round-000029`, is
  producing some game output but reports only `919` checkpoint inputs.
- Latest published rating is also `round-000015` / `919`, while intake is about
  `1888`.
- Gap: the status/control tooling does not yet clearly warn when an active
  game batch is alive but appears to be rating only the old pool. Add that
  guard/readout, then recheck live.
- Validation remains blocked until we see:
  1. an active batch whose checkpoint count is larger than the latest published
     rating when intake is ahead;
  2. a rating snapshot published beyond `919`;
  3. a trainer-facing export from that newer rating;
  4. `trainer-proof` showing running trainers loading that newer export.

Earlier live validation evidence: 2026-05-17 05:35 EDT.

- Deployed read-only run-status app:
  `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py`.
- Added and used deployed-function trainer proof:
  `uv run --extra modal python scripts/curvytron_live_loop_control.py --action trainer-proof --activity-probe-pairs 0 --run-limit 0`.
- Trainer proof result:
  - `136` runs scanned in `9` chunks;
  - `24` target assignment SHAs from generation `4`;
  - `48` runs have a target SHA as the latest applied assignment;
  - `470` assignment refresh applications in the scanned logs;
  - `106820` target env rows;
  - `43723` target provider-ok rows;
  - `0` target provider-false rows.
- Interpretation: the trainer-consumption leg is proven partially. The running
  trainers are not all caught up yet, but the generation-4 tournament export is
  being consumed by real trainer envs.
- Current tournament/rating gap:
  - latest rating remains `round-000015` / `919`;
  - intake has `1763` checkpoints;
  - current internal game-batch artifact `round-000027` has real game output
    and should be allowed to complete/reduce before another drain is spawned.
- Remaining proof:
  1. `round-000027` or the next current batch publishes a rating beyond `919`;
  2. `refresh-if-ready` writes generation `5` or later from that rating;
  3. `trainer-proof` shows trainers applying that newer generation;
  4. survival/ranking signals are measured after the assignment load point.

Earlier live validation evidence: 2026-05-17 05:21 EDT.

- Fixed and redeployed stale current-lane defaults.
- `refresh-if-ready` succeeded against the CZ26 config:
  - generation `4`;
  - rating source `round-000015`;
  - source rows `919`;
  - active rows `100`;
  - rewritten pointers `24`.
- This proves one full path from rating snapshot to trainer-facing assignment
  pointers. Still unproven: running trainers have loaded generation `4`, and
  rating has not caught up to all intake checkpoints yet.
- Current active game-batch artifact is `round-000026` with `1307` checkpoint
  inputs and zero completed game summaries visible in status. The next proof is
  visible game output, rating publish beyond `round-000015`, then another
  refresh.

Earlier live validation evidence: 2026-05-17 05:14 EDT.

- Rating progress is real but partial: latest advanced to `round-000015` /
  `919` checkpoints after the far-ahead blocker patch.
- Trainer-facing refresh is still not proven. `refresh-if-ready` failed because
  deployed current-lane defaults still pointed at old r18fresh config.
- The CZ26 launch manifest has the correct refresh config and `24` refresh
  pointers. The next validation step is to patch shared defaults to CZ26,
  redeploy, then prove trainer refresh advances beyond generation `3`.
- After that, continue rating catch-up and verify the same checkpoints can be
  followed from trainer checkpoint output through intake, tournament rating,
  trainer-facing export, and trainer opponent load.

Earlier live validation evidence: 2026-05-17 04:29 EDT.

- The full loop is still not validated. Active game batch `round-000022` has
  `1361` checkpoint inputs, `659` pairs, and `13839` intended games, but the
  status probe sees `0` completed game summaries.
- Rating/export state is stale: latest rating remains `round-000010` / `588`;
  trainer refresh remains generation `3`.
- New gap: app logs show newer `600s` timeout cancellations after the known
  `1800s` pre-patch timeout wave. Need to identify the function path that still
  has a 10-minute timeout, or prove the batch was launched on old code and let
  safe recovery advance.
- Operator tooling validation: the safe control script now records a drain
  request lease. A repeated `drain-if-ready` call at `08:39:58Z` returned
  `blocked_recent_drain_request`, so it did not stack duplicate drain spawns.
- Operator tooling gap closed enough for now: synchronous drain output explains
  what happened. At `08:43Z`, it showed stale claim repair and spawned rating
  loop `fc-01KRTHMV0DN5TH8SBZM6EM945Y`. Remaining validation is game batch
  creation, game summaries, rating publish, trainer export, and trainer load.

Earlier deployed tooling evidence: 2026-05-17 03:10 EDT.

- New live finding: compact zero-started progress can be stale while game
  workers are still writing outputs. Logs showed `r000012` game outputs after
  the control plane treated that game batch as stale.
- Code patch: the actual intake-drain recovery path now calls
  `_rating_round_skip_decision(..., scan_output_progress=True)`, and a scan
  error blocks silent skipping. Focused local validation passed: `36 passed`.
- Current remote gap: verify `round-000013` after the patch/deploy path. It was
  spawned with `815` checkpoint inputs, which proves the pool-regression fix,
  but not completion, rating reduction, or trainer refresh.
- Follow-up remote status: `round-000013` was skipped and `round-000014` is
  active with `681` checkpoint inputs while intake has `828`. Logs prove
  `r000014` games are completing. The status tool must report this directly
  instead of sending operators to raw logs.
- Local validation after adding the status-tool output probe passed:
  `37 passed`.
- Local validation after splitting full recovery scan from bounded
  `loop-status` activity probe passed: `38 passed`.
- Remote `loop-status` after deploy reports active output directly: intake
  `956`, latest rating `414`, active game batch `round-000015` with `919`
  checkpoint inputs and `21` game summaries found by the bounded probe. Still
  unproven: rating reduction from that batch, trainer-facing export refresh,
  and trainer consumption of the refreshed assignment.
- Follow-up remote `loop-status` contradicted the intended recovery behavior:
  by about `07:07Z`, `round-000015` was marked `skipped` at `07:06:59Z` even
  though the earlier bounded probe saw real output. Latest rating was still
  `414`. This means the full loop is still not validated.
- Current local patch under test: skipped game-batch summaries in
  `loop-status` now include a compact `skip_decision` so the operator readout
  can explain why a batch was skipped without raw Volume spelunking. The next
  live proof must show whether the skip came from an old in-flight drain or
  from a remaining recovery-scan bug.
- Focused local validation for this status-tool patch passed: `39 passed`,
  `17 warnings` from local Modal function execution.

Earlier 2026-05-17 02:02 EDT evidence:

- Focused local suite passed after the compact status tool and timer split:
  `34 passed`.
- Deployed `curvyzero-checkpoint-tournament-v2` with:
  - `--mode loop-status`,
  - duplicate existing-input backoff,
  - separate zero-started game-batch stale timer (`600s`),
  - conservative intake claim stale timer (`24h`).
- Remote `loop-status` worked and showed compact state without raw Volume
  dumps.
- `round-000008` moved from zero-started to complete and wrote ratings.
- Follow-up remote proof: `round-000009` completed, `round-000010` was skipped
  after staying zero-started, and `round-000011` spawned with `681` checkpoint
  inputs and queue length `0`.
- Follow-up remote bug: `round-000012` regressed to a `414`-checkpoint input
  pool while intake had `705`, because continuation mode preferred stale
  `seen_checkpoint_refs`. Patched and tested current-manifest pool selection.
- Superseded remote gap: the next spawned game batch after `round-000012` did
  use the current intake pool. It is `round-000013` with `815` checkpoint
  inputs. The remaining gap is completion/recovery without skipping active game
  output.

Latest operator-tooling update: 2026-05-17 01:55 EDT.

- Added local compact status path `--mode loop-status` for the tournament app.
  This is meant to replace manual raw reads as the default way to answer:
  "Is intake ahead of ratings?", "Is a game batch active or stuck?", and
  "Did trainer refresh advance?"
- Added local duplicate active-game-batch handling: if a caller sees an
  existing input with a different spec, it returns `running_existing_round` with
  phase `existing_input_different_spec` instead of crashing or overwriting.
- Historical before deploy: focused validation was in progress for:
  - compact `loop-status` result shape and zero-started active-batch flag,
  - duplicate existing-input backoff,
  - intake repair regressions.
- Historical before deploy: remote validation still required:
  - run live status against `cz26-live-20260517a`; current safe command is
    `uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 12`,
  - verify whether `round-000008` is truly running or stuck,
  - prove trainer env/provider rows consumed generation `3` assignments.

Latest live-loop repair evidence: 2026-05-17 01:36 EDT.

- Active intake state is clean: exactly one active manifest,
  `manifest:cz26-live-20260517a:elo-cz26-live-20260517a`.
- The stale `all_pairs` live config was found in the active manifest itself,
  not just in old tournaments. The repair now covers Modal Dict load, volume
  load, seed, submit, and subscriber tick artifact writes.
- Current live intake config is bounded:
  `pair_selection=adaptive_v0`, `pairs_per_round=300`,
  `active_pool_limit=100`, `games_per_pair=21`, `games_per_shard=1`.
- Stale recovery patch is locally covered by
  `tests/test_curvytron_checkpoint_intake_repair.py`: `32 passed`.
- Stuck tournament game batch `round-000003` was skipped and cannot block the
  current pipeline. This is not a training round.
- Newer stale zero-progress bounded game batch `round-000005` was also skipped.
- Current bounded tournament game batch `round-000006` exists with `300` pairs
  and `6,300` games. Modal logs show games completing with `ok=true`.
- Checkpoint intake has seen `431+` cz26 checkpoints after the live repair.
- Progress JSON can lag while the game map is running, so live game logs are
  the current running proof. The completion proof is still ratings/latest
  output after reduce.
- Do not use the aggressive 60-second manual claim-recovery setting on the
  current live game batch. Intake claim stale is `24h`; tournament game-batch
  recovery is now the separate `600s` check with real output scanning.
- A sampled trainer already consumed tournament refresh generation `2`
  (`cz26-auto-recipe-07`). The remaining full-loop proof is the next
  generation after the bounded tournament game batch.

Latest remote launch evidence: 2026-05-17 00:31 EDT.

- Full `cz26-full-20260517a` batch submitted: `136/136` rows recorded as
  `spawned`.
- Seed tournament completed: `126/126` games complete, `0` failed games.
- Seed leaderboard publish succeeded: `4` active rows.
- Training-candidate refresh succeeded: `24` refresh pointers rewritten from
  the seed leaderboard.
- All `136` top-level run directories exist in the v2 runs volume.
- Follow-up volume check proved every train call produced at least one
  checkpoint: `136/136`.
- Subscriber automation is proven for current checkpoints: manifest reached
  `140` checkpoints after trainers started, including new `cz26` checkpoints.
- Drain/rating is proven through the first large tournament game batch:
  `round-000002` completed with `903` pairs and `18,963` games.
- `round-000003` was later identified as the bad giant all-pairs tournament
  game batch and skipped. It is not a valid result.
- Currently running: bounded tournament game batch `round-000004`,
  `300` pairs and `6,300` games.
- Not yet fully proven: bounded game-batch completion, post-batch leaderboard
  publish, training-candidate auto-refresh from that larger leaderboard, and
  trainer consumption of the refreshed assignment after the next checkpoint
  boundary.

Latest deployed-loop cleanup: 2026-05-17 00:11 EDT. Result:
focused patch suite `15 passed`; broader loop/tournament/provider suite
`338 passed, 14 skipped`.

```bash
uv run --extra dev --extra modal pytest -q \
  tests/test_curvytron_checkpoint_tournament.py::test_rating_writer_finished_trusts_completed_latest_over_stale_root_progress \
  tests/test_curvytron_checkpoint_tournament.py::test_rating_loop_start_state_skips_repaired_orphan_round \
  tests/test_curvytron_shared_contracts.py \
  tests/test_curvytron_next_batch_manifest.py

uv run --extra dev --extra modal pytest -q \
  tests/test_curvytron_checkpoint_tournament.py \
  tests/test_curvytron_shared_contracts.py \
  tests/test_curvytron_next_batch_manifest.py \
  tests/test_curvytron_training_candidate_controller_local.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_opponent_leaderboard.py \
  tests/test_opponent_mixture.py
```

Concrete bugs found by deployed canary:

- Static intake bug: reseeding with seed refs plus a run prefix did not upgrade
  the manifest to live-watch. The canary status now shows
  `explicit_refs_plus_run_watch`, `run_id_prefix=cz26c-`, and six checkpoint
  refs including the new canary checkpoints.
- Stale progress bug: `latest.json` had completed `round-000009`, but the root
  `progress.json` still said that same round was running. The drain treated the
  rating writer as unfinished and refused to continue. `_rating_writer_has_finished`
  now trusts a completed latest snapshot over stale progress for the same or
  older round.
- CLI config-ref bug: manual canary auto-refresh can now pass
  `training_candidate_refresh_config_ref` explicitly.

Remote canary evidence so far:

- `intake-status`: live-watch manifest, `checkpoint_count=6`, queue initially
  `6`.
- `intake-drain --detach`: drained six events, repaired stale claim, spawned
  rating continuation `fc-01KRT21JW6AFJ0T8E883VMTN5W`.
- `round-000010/input.json`: `6` checkpoints, `15` pairs, `315` games, includes
  both current `cz26c` checkpoints.

Still required before full batch:

- `round-000010` must finish and write `latest.json` with six rows.
- Publish canary training leaderboard from that latest snapshot.
- Run `training-candidate-auto-refresh` using the canary config ref.
- Prove the canary trainer consumed the same assignment SHA in refresh/provider
  rows.

Latest focused contract cleanup: 2026-05-16 21:45 EDT. Result:
`22 passed` across the new manifest/name/submitter checks and the existing
shared refresh/publish path.

```bash
uv run --extra dev --extra modal pytest -q \
  tests/test_curvytron_naming.py \
  tests/test_curvytron_next_batch_manifest.py \
  tests/test_curvytron_survivaldiag_submitter.py

uv run --extra dev --extra modal pytest -q \
  tests/test_curvytron_shared_contracts.py \
  tests/test_curvytron_training_candidate_controller_local.py::test_training_candidate_refresh_writes_leaderboard_and_rewrites_control_pointer \
  tests/test_curvytron_training_candidate_controller_local.py::test_checkpoint_intake_rating_leaderboard_assignment_trainer_lineage_chain \
  tests/test_curvytron_checkpoint_tournament.py::test_opponent_leaderboard_publish_writes_snapshot_latest_and_pointer \
  tests/test_curvytron_checkpoint_tournament.py::test_build_pair_specs_defaults_to_unordered_no_self_pairs \
  tests/test_curvytron_checkpoint_tournament.py::test_rating_snapshot_uses_batch_elo_and_is_order_stable
```

This proves the cleaned `cz26` manifest contract locally. It still does not
replace the deployed canary proof.

Broader focused suite rerun: 2026-05-16 21:45 EDT. Result:
`448 passed, 24 skipped`. The first run exposed a missing internal default for
`reward_outcome_alpha` in `_run_visual_survival_train`; that contract was fixed
and the same suite reran green.

```bash
uv run --extra dev --extra modal pytest -q \
  tests/test_curvytron_shared_contracts.py \
  tests/test_curvytron_naming.py \
  tests/test_curvytron_next_batch_manifest.py \
  tests/test_curvytron_survivaldiag_submitter.py \
  tests/test_curvytron_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_intake_repair.py \
  tests/test_curvytron_training_candidate_controller_local.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_source_state_visual_survival_learner_seat_regression.py \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py \
  tests/test_opponent_mixture.py \
  tests/test_opponent_leaderboard.py \
  tests/test_curvytron_gif_browser.py
```

Last run: 2026-05-16. Result: `45 passed in 2.47s`.

```bash
uv run --extra dev --extra modal pytest -q \
  tests/test_curvytron_shared_contracts.py \
  tests/test_run_management.py \
  tests/test_curvytron_tournament_scheduler_guardrails.py \
  tests/test_curvytron_tournament_scheduler_fairness.py \
  tests/test_opponent_leaderboard.py \
  tests/test_materialize_curvytron_leaderboard_assignment.py \
  tests/test_source_state_visual_survival_learner_seat_regression.py
```

## Targeted Full-Loop Pieces

Last run: 2026-05-16. Result: `14 passed in 1.16s`.
Warnings: three expected local Modal warnings about functions executing locally
without mounted remote data.

```bash
uv run --extra dev --extra modal pytest -q \
  tests/test_curvytron_checkpoint_tournament.py::test_build_pair_specs_defaults_to_unordered_no_self_pairs \
  tests/test_curvytron_checkpoint_tournament.py::test_adaptive_v0_pair_specs_are_budgeted_unique_and_tagged \
  tests/test_curvytron_checkpoint_tournament.py::test_rating_snapshot_uses_batch_elo_and_is_order_stable \
  tests/test_curvytron_checkpoint_tournament.py::test_opponent_leaderboard_publish_writes_snapshot_latest_and_pointer \
  tests/test_curvytron_checkpoint_intake_repair.py::test_live_intake_rating_defaults_are_bounded_adaptive \
  tests/test_curvytron_checkpoint_intake_repair.py::test_live_watch_drain_continues_past_partial_existing_rating_and_old_claim \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_progress_writer_updates_browser_speed_file \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_own_checkpoint_opponent_refresh_writes_pointer_and_assignment \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_refresh_hook_handles_due_unchanged_and_failure \
  tests/test_curvytron_training_candidate_controller_local.py::test_training_candidate_refresh_writes_leaderboard_and_rewrites_control_pointer \
  tests/test_source_state_visual_survival_learner_seat_regression.py::test_random_learner_seat_mode_uses_both_seats_deterministically_with_dynamic_seed \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_visual_survival_opponent_immortal_blocks_opponent_wall_death \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_background_eval_inspection_and_gif_can_be_explicitly_enabled \
  tests/test_curvytron_gif_browser.py::test_default_browser_lists_only_runs_with_picker_flag
```

## Reorientation Smoke

Last run: 2026-05-16 16:23 EDT. Result: `8 passed in 1.15s`.

```bash
uv run --extra dev --extra modal pytest -q \
  tests/test_curvytron_training_candidate_controller_local.py::test_checkpoint_intake_rating_leaderboard_assignment_trainer_lineage_chain \
  tests/test_source_state_visual_survival_learner_seat_regression.py::test_random_learner_seat_mode_uses_both_seats_deterministically_with_dynamic_seed \
  tests/test_curvytron_shared_contracts.py
```

This confirms the current local proof chain and constants still line up after
the latest doc/code changes. It is not the deployed Modal canary.

## Untested Gaps

- No single local pytest closes the whole loop in one synthetic scenario:
  fake checkpoint -> intake -> bounded rating -> leaderboard publish ->
  training-candidate refresh -> trainer refresh consumption.
- `loop-status` is a compact health readout, not full lineage. It summarizes
  the live state; it does not by itself prove a specific checkpoint crossed
  every boundary.
- Checkpoint production tests are hook-level; they do not run a real LightZero
  learner long enough to produce an iteration checkpoint and all sidecars.
- Modal lifecycle/fanout is mostly mocked locally. Child survival after parent
  command exit remains remote-smoke territory.
- Random learner seat plus refreshed frozen/provider opponent is not tested as
  one combined contract.
- Persisted live artifacts can still carry old no-GIF config even when source
  defaults are fixed. Source tests do not prove an old arena was reseeded.

## Immediate Validation Actions

1. Run the fast local sweep.
2. Run targeted full-loop pieces if the fast sweep is clean.
3. Add a local synthetic full-loop test before the next major refactor.
4. For deployed proof, verify actual assignment SHA consumption and env
   provider-load rows, not just scheduled Modal calls.

## Observability Test Gap

Add tests around the lineage/health artifacts before relying on them:

- A fake checkpoint write emits `checkpoint_written` with checkpoint ref,
  iteration, metadata presence, size, and mtime.
- A fake intake submit/tick/drain emits seen/enqueued/claim/spawn events without
  blocking the underlying pipeline when lineage append fails.
- A fake rating round emits start/reduced/latest-written events with pair/game
  counts and stable/max-delta fields.
- A fake leaderboard publish/training-candidate refresh emits export generation,
  selected refs, assignment sha, and pointer rewrite rows.
- A fake trainer refresh emits loaded/applied rows and distinguishes unchanged,
  applied, and failed-after-reset decisions.

The first synthetic full-loop test should assert the stages appear in order for
one checkpoint and one assignment SHA.
