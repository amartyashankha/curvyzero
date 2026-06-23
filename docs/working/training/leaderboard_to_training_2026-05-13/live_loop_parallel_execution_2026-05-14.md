# Live Loop Parallel Execution - 2026-05-14

Historical note: this page contains invalidated `v2refresh18*` launch details.
Its `opponent_assignment_refresh_interval_train_iter=50` examples are history,
not the current restart18 default. Current restart manifests use the shared
all-v2 contract and default to refresh interval `2000`.

## North Star

Prove the real loop, not just busy pieces of it:

1. Start trainers from the current leaderboard champion.
2. Trainers write exact immutable checkpoints.
3. Intake discovers those checkpoints.
4. The live arena rates them as a continuation, not a reset.
5. A clean final rating publishes a public leaderboard snapshot.
6. Coach materializes a new immutable assignment from that snapshot.
7. A trainer consumes that assignment, either at launch or through a proven
   refresh boundary.

The current goal is a truthful proof chain. Do not call the loop closed until
the assignment comes back into a trainer and the artifacts say so.

## Exact Live Names

- Champion-started training batch: `curvy-champion-canary18-20260514a`
- Champion-started trainer run prefix: `curvy-champ18a`
- Current live arena label: `champ18a`
- Direct refresh canary: `refresh-direct-canary-20260514a`
- Assignment-bank run: `curvy-champ18a-assignments`
- Assignment-bank attempt: `try-champ18a-assignments`

Current facts:

- 2026-05-14 evening correction: prepare a v2 Volume lane now. The old
  `curvyzero-runs` Volume is write-fragile (`too many layers in volume`) and
  near its inode limit. New real runs should use `curvyzero-runs-v2` and
  `curvyzero-curvytron-tournaments-v2`.
- Fresh v2 relaunch `curvy-v2champ18b-20260514a` is running as the current
  trainer/checkpoint producer. It was submitted at about 2026-05-14 21:33 EDT
  with 18 train calls and 18 pollers. First artifact pass found all 18 run
  manifests and all 18 `progress_latest.json` refs present in
  `curvyzero-runs-v2`.
- `curvy-v2champ18b-20260514a` is not a full-loop proof by itself. The rows use
  static launch-time assignments. The missing proof is still: v2 checkpoint
  intake, tournament continuation, public leaderboard publication, assignment
  materialization, and trainer consumption/refresh from that new assignment.
- Aggressive full-loop attempt is now in flight:
  - tournament: `curvy-v2champ18-live-20260514a`;
  - rating: `elo-v2champ18-live-20260514a`;
  - seeded from exact `curvy-v2champ18b-*` run ids, not a prefix;
  - `intake-seed` found 18/18 fresh v2b checkpoint refs, all at
    `iteration_0.pth.tar`, queued 18 events, and spawned rating call
    `fc-01KRMMEMG81BEZYVHNJB9T165P`;
  - status/tick found the manifest active in
    `curvyzero-curvytron-checkpoint-intake-v2`, queue drained, 18 seen refs,
    zero missing refs;
  - round 0 is all-pairs: 153 pairs, 3213 games, one-frame, GIFs enabled;
  - first progress read showed `status=running`, 66 started pairs, estimated
    1386 seen games, zero failed games.
- This proves the trainer-to-intake handoff exists for v2b. It still does not
  prove the back half until the rating writes final artifacts, the controller
  publishes/materializes, and a trainer consumes the resulting assignment.
- Parallel small proof lane:
  - tournament: `curvy-v2tiny-loop-20260514a`;
  - rating: `elo-v2tiny-loop-20260514a`;
  - seeded from the first four exact v2b run ids;
  - found 4/4 `iteration_0.pth.tar` refs, queued 4 events, and created rating
    artifacts in `curvyzero-curvytron-tournaments-v2`;
  - progress read showed 6 pairs, 126 games, 5 started pairs, estimated
    105 seen games, zero failed games.
- The small lane is not a replacement for the 18-row proof. It is a fast path
  for publish/materialize/smoke while the larger all-pairs round finishes.
- A same-id relaunch of the 18-row champion batch is not a clean new proof. At
  least one row failed correctly because `initial_policy_checkpoint_ref` was
  set while same-run auto-resume found an existing checkpoint. Fresh run ids and
  fresh attempt ids are mandatory for the next real launch.
- Read-only old-batch health check: 18/18 rows started and wrote artifacts;
  8 were still running and 10 had failed. Survival/eval means are noisy rather
  than a clean upward curve. This is useful evidence, not a green light to keep
  reusing the same old batch ids.
- Cleanup action: stopped stale failed app `ap-5CRdftJaDAF5LUxT9iw2C6`.
- `champ18a` round 0: 4-checkpoint all-pairs completed `126/126` games with
  zero failures.
- `champ18a` round 1: completed cleanly for 8 checkpoints: 28 pairs, 588 games,
  zero failures, one-frame, GIFs enabled. This is a small mechanics proof, not
  the final 18-run proof.
- `champ18a` round 2: spawned as a continuation for the larger 28-checkpoint
  pool after stale-claim repair. First progress read showed 971/7938 games
  complete, 50 pairs started, and zero failures.
- Small downstream canary from round 1 is in flight:
  - published immutable leaderboard snapshot
    `champ18a-r1-small8-20260514a`;
  - materialized `stable_slots_v1` assignment
    `champ18a-r1-small8-assignment-20260514a`;
  - wrote assignment to the training Volume;
  - tiny trainer smoke
    `champ18a-small8-assignment-consume-smoke-20260514a` passed.
- Small downstream canary evidence:
  - public snapshot sha:
    `3fe61bb2062b017f7f7bf6846a67c95bf3b653cb6b219fc558e9660189dc6f8d`;
  - written assignment ref:
    `training/lightzero-curvytron-visual-survival/curvy-champ18a-live-assignment-small8-20260514a/attempts/try-curvy-champ18a-live-assignment-small8-20260514a/opponents/assignments/champ18a-r1-small8-assignment-20260514a/assignment.json`;
  - written assignment sha:
    `cec4cc526313c11ee38378b42c67f802231ae19588781fd5fc12f8bd6afc8127`;
  - trainer summary says `ok=true`, no problems, and champion bootstrap
    `loaded=true` into `model` and `target_model`;
  - 443 env telemetry rows used the exact assignment ref/hash with
    `opponent_provider_load_ok=true`.
- `refresh-direct-canary-20260514a` passed as a trainer-side direct assignment
  refresh proof. It applied the second immutable assignment at a refresh
  boundary and later env telemetry used the new assignment hash.
- Pointer upload is currently blocked by Modal Volume `too many layers`.
- The 18 `champ18a` trainers have static assignments only. They cannot prove
  running-trainer refresh.

## Parallel Lanes

| Lane | What it proves | What it cannot prove |
| --- | --- | --- |
| `curvy-champion-canary18-20260514a` trainers | A champion-started 18-row batch can launch, train, write checkpoints, and use immutable launch-time assignments. | It cannot prove automatic opponent refresh because the rows have static assignments only. |
| `champ18a` arena round 0 | The arena can complete a small all-pairs baseline with clean artifacts: 4 checkpoints, `126/126` games, zero failures. | It does not prove continuation, public publish, materialization, or trainer consumption of a new assignment. |
| `champ18a` arena round 1 small publish canary | The arena can complete a clean continuation, publish a public leaderboard, materialize an assignment, write it, and run a trainer that actually consumes it. | It is only 8 checkpoints, so it is not final strength evidence and does not cover all 18 trainer runs. |
| `champ18a` arena round 2 | The same arena can continue into the larger 28-checkpoint pool. | It is not publishable until it finishes with matching artifacts and zero failures. |
| `refresh-direct-canary-20260514a` | A running trainer can switch from one immutable assignment to another at a safe refresh boundary. Artifacts show `decision=applied`, the new assignment hash, and later env telemetry using the new assignment. | It does not prove the full leaderboard publisher/materializer/controller path. It proves the trainer-side refresh mechanism only. |
| Pointer upload / Volume recovery | The operator path can write the small mutable pointer needed by the refresh canary despite Modal Volume layer pressure. | It does not prove learning quality, rating correctness, or assignment consumption by itself. |
| `curvy-v2champ18b-20260514a` trainers | Fresh v2 app/Volume launch, champion bootstrap, assignment read, and checkpoint production under fresh ids. | It does not prove automatic tournament intake, leaderboard publication, assignment materialization, or running-trainer refresh because its assignment refs are static. |
| `curvy-v2tiny-loop-20260514a` arena | Fast back-half v2 proof lane. Four fresh v2b checkpoint refs were seeded into a v2 tournament. Round 0 completed 6/6 pairs and 126/126 games, zero failures, `stable=true`, ratings written. Promotion/materialization/smoke is now running with strict non-provisional gates. | It is only four initial checkpoints, so it is a mechanics proof, not strength evidence. |
| `curvy-v2champ18-live-20260514a` arena | Full 18-way v2 all-pairs lane from the fresh v2b initial checkpoints. All 153 pairs / 3213 games were seen by lightweight progress with zero failures; reducer/final summary artifacts are still being monitored. | Not publishable until final round artifacts exist and agree. |

## Artifact Pass Gates

Before treating a lane as passed, require artifacts rather than log optimism:

- Round input, progress, results, ratings, and latest files exist for the exact
  round being claimed.
- Checkpoint roster, pair count, game count, round index, and context hash agree
  across those files.
- Failed games are zero, unless a human explicitly accepts and documents the
  failure.
- Continuation rounds are labeled as later rounds and preserve prior rating
  state; they must not silently restart as round 0.
- Public leaderboard publish uses the immutable final rating snapshot, not a
  mutable latest pointer.
- Assignment materialization records the source snapshot ref/hash and writes an
  immutable `assignment.json` plus audit.
- Trainer consumption is proven by trainer artifacts: assignment ref/hash in
  telemetry and `opponent_provider_load_ok=true`.
- Refresh is proven only by `opponent_assignment_refresh_events.jsonl` showing
  an applied new assignment and later `env_steps.jsonl` rows using that new
  assignment ref/hash.

## Operating Rule

Run independent proof lanes in parallel. If a later dependent step might break,
still run it speculatively in parallel when it is safe and cheap to do so, then
discard the output if an upstream gate fails.

Examples:

- Let `champ18a` round 1 continue while the pointer upload blocker is being
  resolved.
- Keep `refresh-direct-canary-20260514a` moving because it proves the missing
  trainer-refresh contract that the 18 static `champ18a` trainers cannot prove.
- If round 1 finishes but fails artifact gates, discard any publish or
  assignment derived from it and relaunch from the last clean source.
- If the refresh canary applies a pointer before the pointer target is known
  good, keep the telemetry as a trainer-mechanics smoke only; do not promote the
  assignment as leaderboard proof.

The discipline is simple: parallelize for learning speed, gate promotion on
artifacts, and label every partial proof by what it actually proves.

## 2026-05-14 Late Live State

- Tiny v2 tournament:
  - tournament: `curvy-v2tiny-loop-20260514a`;
  - rating: `elo-v2tiny-loop-20260514a`;
  - round 0 is complete: 4 checkpoints, 6 pairs, 126 games, zero failures,
    `stable=true`, `phase=ratings_written`.
  - controller promotion is in progress with:
    `active_min_distinct_opponents=3`, `active_min_valid_games=63`,
    `max_active_rank=4`, and `--run-smoke`.
- Full v2 tournament:
  - tournament: `curvy-v2champ18-live-20260514a`;
  - rating: `elo-v2champ18-live-20260514a`;
  - seeded 18 exact v2b run ids and found 18/18 current checkpoint refs;
  - all 153 pair tasks have started and lightweight progress has seen all
    3213 expected games with zero failures;
  - wait for final reducer artifacts before publish/materialize.
- Training batch:
  - `curvy-v2champ18b-20260514a` remains alive and useful for producing
    checkpoints, but it is static-assignment only. A new refresh-enabled run is
    required to prove leaderboard-derived assignments come back into an already
    running trainer.

## 2026-05-14 Promotion / Refresh Update

- Tiny v2 lane promotion passed:
  - published `v2tiny-loop-20260514a / v2tiny-loop-r0-20260514a`;
  - 4 active rows, 0 provisional rows;
  - wrote assignment
    `training/lightzero-curvytron-visual-survival/v2tiny-loop-assignment-bank-20260514a/attempts/try-v2tiny-loop-assignment-bank-20260514a/opponents/assignments/v2tiny-loop-r0-assignment-20260514a/assignment.json`;
  - smoke trainer loaded the assignment, ran MuZero, and verified
    `opponent_provider_load_ok=true` for all telemetry rows.
- Full 18-way v2 lane promotion passed:
  - source: 18 checkpoints, 153 pairs, 3213 games, zero failures,
    one-frame settings;
  - published `v2champ18-live-20260514a / v2champ18-live-r0-20260514a`;
  - 18 active rows, 0 provisional rows;
  - wrote assignment
    `training/lightzero-curvytron-visual-survival/v2champ18-live-assignment-bank-20260514a/attempts/try-v2champ18-live-assignment-bank-20260514a/opponents/assignments/v2champ18-live-r0-assignment-20260514a/assignment.json`;
  - smoke trainer loaded the assignment, ran MuZero, and verified
    `opponent_provider_load_ok=true` for all telemetry rows.
- Refresh pointer canary:
  - first attempt was launched without `--wait-for-train`, so the ephemeral app
    stopped before refresh events were visible. Treat it as a launch mistake,
    not a refresh proof.
- Ambitious full refresh batch:
  - matrix: `curvy-v2refresh18-20260514a`;
  - submitted 18 H100 rows and 18 pollers through the v2 deployed train app;
  - launch assignments are the three recipe assignments built from the full
    18-way leaderboard snapshot;
  - every train row has
    `opponent_assignment_refresh_interval_train_iter=50`;
  - every train row reads the shared refresh pointer:
    `training/lightzero-curvytron-visual-survival/v2refresh18-control-20260514a/attempts/try-v2refresh18-control-20260514a/opponents/refresh_pointer.json`;
  - the pointer currently targets the full 18-way assignment above. Since the
    launch assignments are recipe-specific, a successful refresh should log an
    `applied` event and switch the env telemetry to the full assignment hash.

## 2026-05-14 Late Refresh18 Proof State

- `curvy-v2refresh18-20260514a` is the first real refresh-enabled 18-row batch.
  All 18 rows use the same shared pointer:
  `training/lightzero-curvytron-visual-survival/v2refresh18-control-20260514a/attempts/try-v2refresh18-control-20260514a/opponents/refresh_pointer.json`.
  Every row has `opponent_assignment_refresh_interval_train_iter=50`.
- Trainer-side refresh is proven at least for row `r001`:
  - `opponent_assignment_refresh_events.jsonl` contains `decision=applied`;
  - it applied the full 18-way assignment sha
    `d881126f31b726b52a1e932b42b3eb3734acbd0e51faef78a8ef7a8b151155e6`;
  - env readiness reported `ok=true` for 256 envs;
  - downloaded env telemetry had 41,040 rows with the refreshed assignment
    hash, 37,165 of those with `opponent_provider_load_ok=true`, and only 130
    launch-assignment rows before the refresh.
- Important caveat: the refresh event also recorded
  `volume_reload.ok=false` because the current working directory was inside the
  mounted Volume. The assignment still resolved and envs loaded it. Treat this
  as a freshness footgun to keep watching, not as a failed current refresh.
- The refresh18 intake/tournament lane is active:
  - tournament: `curvy-v2refresh18-live-20260514a`;
  - rating: `elo-v2refresh18-live-20260514a`;
  - initial seed started too early and first rated only 8 checkpoints;
  - a later `intake-status`/`intake-tick` repaired the manifest to 18/18
    checkpoints, zero missing, queue empty.
- Round `round-000000` completed for the first 8 checkpoints:
  28 pairs, 588 games, zero failures, ratings written.
- Round `round-000001` completed the 18-checkpoint all-pairs tournament:
  153 pairs, 3213 games, zero failures. A detailed game-summary progress read
  counted all 3213 games.
- Current blocker before closing the feedback loop: `ratings.json` and
  `results.json` exist for `round-000001`, but `latest.json` still pointed at
  `round-000000` when the promotion controller first ran. The controller
  correctly refused to publish. A waited reduce/finalize repair is in progress;
  only retry promotion after `latest.json` points at `round-000001`.
- Next required proof:
  1. publish/materialize/write a new assignment from `round-000001`;
  2. update the shared refresh pointer to that new assignment;
  3. observe at least one still-running trainer log a second `decision=applied`
     refresh event with a new assignment hash;
  4. observe later env telemetry using that new hash.

## 2026-05-14 22:10 EDT Incident: Stale Latest Reused a Round

- Do not promote `curvy-v2refresh18-live-20260514a /
  elo-v2refresh18-live-20260514a` as proof.
- What happened:
  - `round-000001/ratings.json` and `results.json` existed for the clean
    18-checkpoint result: 153 pairs, 3213 games, zero failures.
  - `latest.json` stayed on `round-000000`.
  - A later continuation trusted stale `latest.json`, reused `round-000001`,
    and overwrote `input.json`/`progress.json` with a larger 20-checkpoint
    round: 190 pairs, 3990 games.
- The promotion controller correctly refused the first stale latest. After a
  manual pointer copy, the artifacts still disagreed, so this lane is now
  contaminated evidence.
- Root cause to fix: continuation must not pick only from mutable `latest.json`,
  and a rating round must refuse to overwrite a round directory that already has
  completed `ratings.json` or `results.json`.
- Parallel patch lane: add regression tests and the smallest guard for this
  stale-latest/round-overwrite failure. Do not deploy that patch until tested.

## 2026-05-14 22:11 EDT Clean Proof Lane

- Started a fresh proof arena/rating id from explicit current trainer run ids:
  - tournament: `curvy-v2refresh18-proof-20260514b`;
  - rating: `elo-v2refresh18-proof-20260514b`.
- Launch command used `modal run --detach`, `--mode rating`, no continuation,
  latest checkpoint selection, expected checkpoint count 18, all-pairs, 21 games
  per pair, one-frame, GIFs on with 5 samples per pair.
- First progress read:
  - found 18 checkpoints, zero missing;
  - planned 153 pairs / 3213 games;
  - started running immediately.
- Intended use: once complete, promote this clean round into a new immutable
  assignment, update the shared refresh pointer, then verify still-running
  trainers log a second applied refresh with the new assignment hash.

## 2026-05-14 22:14 EDT Trainer Audit

- All 18 `curvy-v2refresh18-20260514a` trainer rows still report running.
- 15/18 rows have produced `iteration_10000.pth.tar`; `r006`, `r013`, and
  `r016` still only show `iteration_0`.
- 15/18 rows have later `unchanged` refresh events against the original pointer
  assignment sha `d881126f31b7...`. This is expected until the shared refresh
  pointer is updated to a new assignment.
- Survival/length signal is not yet clearly positive. A quick eval-manifest
  comparison of `iteration_0` vs `iteration_10000` showed 4 rows improved mean
  steps, 11 worsened, and 3 missing the later eval. Treat this as early/noisy;
  do not claim learning progress yet.

## 2026-05-14 22:18 EDT Proof Assignment Published and Pointer Updated

- Clean proof lane completed and passed promotion:
  - tournament/rating:
    `curvy-v2refresh18-proof-20260514b /
    elo-v2refresh18-proof-20260514b`;
  - round: `round-000000`;
  - 18 checkpoints, 153 pairs, 3213 games, zero failures;
  - leaderboard snapshot:
    `tournaments/curvytron/leaderboards/v2refresh18-proof-20260514b/snapshots/v2refresh18-proof-r0-20260514b.json`;
  - active rows: 18; provisional rows: 0.
- The assignment was materialized and written:
  - ref:
    `training/lightzero-curvytron-visual-survival/v2refresh18-proof-assignment-bank-20260514b/attempts/try-v2refresh18-proof-assignment-bank-20260514b/opponents/assignments/v2refresh18-proof-r0-assignment-20260514b/assignment.json`;
  - sha:
    `6b8273d632d63a08d4dce44ab16329ff3e2e7aeb31082237960a83f2d7023e07`.
- Trainer smoke passed from that assignment:
  - `ok=true`;
  - 335 env telemetry rows;
  - all 335 provider rows ok;
  - champion bootstrap loaded the rank-1 `iteration_10000` checkpoint.
- The shared refresh pointer was overwritten and verified at:
  `training/lightzero-curvytron-visual-survival/v2refresh18-control-20260514a/attempts/try-v2refresh18-control-20260514a/opponents/refresh_pointer.json`.
- Next gate: observe a still-running `curvy-v2refresh18-20260514a` trainer log a
  second `decision=applied` refresh with assignment sha
  `6b8273d632d63a08d4dce44ab16329ff3e2e7aeb31082237960a83f2d7023e07`, then
  observe later env telemetry using that sha.

## Simplest Robust Controller Shape

The production-shaped operator should become one small controller around
existing pieces:

1. Poll the exact live tournament round.
2. Fetch round input, progress, results, ratings, and latest artifacts.
3. Refuse publish if roster, pair count, game count, round id, or failure count
   disagree.
4. Publish one immutable public leaderboard snapshot.
5. Materialize one immutable `stable_slots_v1` assignment from that snapshot.
6. Write the assignment and audit to the training Volume.
7. Run one tiny trainer smoke from that assignment or refresh to it.
8. Record a decision log with snapshot ref/hash, assignment ref/hash, and smoke
   result.

Keep this boring. Volume JSON is durable truth. Modal Dict and Queue are only
wakeups or pointers. The trainer must not read Elo, Queue, or leaderboard rows
while learning.

## 2026-05-14 Late Control-Volume Fix

- The first reload fix moved cwd out of `/runs`, but the remote smoke showed a
  second real blocker: LightZero keeps a TensorBoard event file open under the
  training Volume, so reloading that same Volume while training is not a
  reliable live-control path.
- Decision: keep trainer logs/checkpoints on `curvyzero-runs-v2`, and move live
  opponent refresh pointer plus refreshed assignment reads to a separate quiet
  Volume, `curvyzero-curvytron-control-v2`, mounted at `/control`.
- Do not count a stale read as `unchanged`: pointer-backed refresh now requires
  a successful control Volume reload. If reload fails, the trainer keeps the
  previous assignment and records a failure/kept-previous event.
- Assignment writes can now target the control Volume and mirror selected frozen
  checkpoint files into `/control/opponent-checkpoints/...`, so refreshed
  assignments do not require the running trainer to reload the hot training
  Volume to see newly selected opponent files.
- Next gates: deploy trainer, write the proof assignment to control, upload the
  control pointer, launch the fresh 18-row patched batch, then observe an actual
  in-running assignment switch from the control pointer.

## 2026-05-14 22:41 EDT Patched Control-Volume Launch

- Root cause of the failed live refresh proof: running trainers could not reliably reload the hot training Volume because LightZero keeps TensorBoard/event files open under `/runs`.
- Implemented the simple fix: live refresh reads now use separate quiet Volume `curvyzero-curvytron-control-v2` mounted at `/control`; `/runs` remains for checkpoints/logs.
- Pointer-backed refresh now fails closed if the control Volume reload cannot prove freshness.
- Assignment writes can target `/control` and mirror selected frozen checkpoint files into `/control/opponent-checkpoints/...`.
- Local validation before launch: `tests/test_curvytron_live_checkpoint_eval_plumbing.py` passed 91 tests / 3 skipped; launch-focused tests passed; ruff passed.
- Deployed `curvyzero-lightzero-curvytron-visual-survival-train-v2` with these changes.
- Wrote refresh target assignment to control:
  - ref: `training/lightzero-curvytron-visual-survival/v2refresh18p-control-20260514b/attempts/try-v2refresh18p-control-20260514b/opponents/assignments/v2refresh18-proof-r0-assignment-20260514b/assignment.json`
  - sha: `9c22f23b2dce3c72b9b944cb9710bcc97e407ab40f3671ba6d70afb3c794c9eb`
  - mirrored frozen checkpoints: 2.
- Uploaded shared refresh pointer to control:
  - `training/lightzero-curvytron-visual-survival/v2refresh18p-control-20260514b/attempts/try-v2refresh18p-control-20260514b/opponents/refresh_pointer.json`.
- Launching patched 18-row batch from manifest `curvy-v2refresh18p-20260514b`; every row has refresh interval 50 and the shared control pointer.

Next proof gates:

1. All 18 jobs write run/attempt/progress artifacts.
2. Refresh events show `decision=applied`, control reload `ok=true`, assignment sha `9c22f23b...`.
3. Later env telemetry uses that assignment sha with provider load ok.
4. Checkpoints appear, then the tournament/intake side can rate them and publish a newer assignment for another pointer update.

## 2026-05-14 22:44 EDT Patched Batch Startup Proof

- `curvy-v2refresh18p-20260514b` submitted successfully.
- Submission record: `artifacts/local/curvytron_tonight18_manifests/curvy-v2refresh18p-20260514b/submission-full-control.json`.
- 18/18 train calls spawned and 18/18 pollers spawned in the single deployed app.
- 3/3 launch recipe assignments were written to the control Volume with mirrored checkpoint refs.
- 18/18 run manifests exist on `curvyzero-runs-v2`.
- 18/18 rows wrote `opponent_assignment_refresh_events.jsonl`.
- 18/18 rows applied the shared control assignment with sha `9c22f23b2dce3c72b9b944cb9710bcc97e407ab40f3671ba6d70afb3c794c9eb`.
- 18/18 refresh events reported control Volume reload `ok=true`.

This proves the old `/runs` reload bug is fixed for this batch: the trainers can read a changed control pointer from `/control` and reset their opponent assignment. Still pending: prove later env telemetry rows use the same sha, wait for real checkpoints, feed those checkpoints to tournament/intake, publish the next assignment, update the pointer again, and observe a second assignment switch.

## 2026-05-14 22:48 EDT Env Telemetry Proof

Remote Volume-side inspection passed for `curvy-v2refresh18p-20260514b`:

- 18/18 `env_steps.jsonl` files exist.
- 18/18 files contain rows with refreshed assignment sha `9c22f23b2dce3c72b9b944cb9710bcc97e407ab40f3671ba6d70afb3c794c9eb`.
- 18/18 files contain provider-ok rows for that sha.
- 18/18 files report `max_refresh_index=1`.
- Parse errors: 0 in the remote inspector.

This proves the patched trainers are actually using the refreshed control-volume assignment in environment self-play. It does not yet prove the next tournament feedback cycle. Next gate: wait for checkpoints from this patched batch, feed them into the tournament/intake lane, publish/materialize the next assignment, update the same control pointer, and watch at least one still-running trainer apply refresh index 2.

## 2026-05-14 22:52 EDT Current Parallel Lanes

Main lane:

1. Keep the 18 patched trainers running.
2. Keep the live tournament watch running:
   `curvy-v2refresh18p-live-20260514b /
   elo-v2refresh18p-live-20260514b`.
3. Wait for either:
   - the initial `iteration_0` tournament round to finish, so it can publish a
     first clean leaderboard for this lane; or
   - the trainers to produce `iteration_10000` checkpoints, so intake can prove
     the real future-checkpoint path.

Status snapshot:

- The live watch seed command completed.
- Intake found 18/18 patched trainer checkpoints and enqueued 18 refs.
- First rating round planned 153 pairs / 3213 games.
- First progress read: 27 pairs started, no known failures, no completed shard
  summaries yet.
- Checkpoint inspector: 18/18 rows have `iteration_0`; 0/18 rows have
  `iteration_10000`.

Resolved critique from Sagan:

- The refresh pointer is no longer missing.
- The stale top-level manifest pointer is no longer stale.
- Pointer and row refs now agree on
  `v2refresh18p-control-20260514b/.../refresh_pointer.json`.

Operating rule for the next wait:

- Do not sit idle on one blocking condition. Poll tournament progress and
  checkpoint arrival separately.
- If the full live lane is slow, run a smaller canary in parallel only if it
  exercises the same real interfaces. Do not invent a separate fake path.
- Keep every claim tied to a concrete artifact: checkpoint refs, progress JSON,
  rating `latest.json`, public leaderboard snapshot, assignment ref/hash,
  pointer ref/hash, and env telemetry refresh index.

## 2026-05-14 22:55 EDT Red-Team Checklist

Before calling this loop complete, require all of these:

1. At least one checkpoint produced after the patched trainers started, not only
   `iteration_0`.
2. Intake status shows that new checkpoint accepted into
   `curvy-v2refresh18p-live-20260514b`.
3. A rating round that includes the new checkpoint completes with final
   `results.json`, `ratings.json`, and `latest.json`.
4. Round artifacts agree on roster/context, pair count, game count, and zero
   failures.
5. Promotion/materialization writes a new control-volume assignment and records
   the control sha after checkpoint refs are mirrored.
6. The shared control pointer is updated only after the assignment is present.
7. At least one still-running trainer logs refresh index 2 with
   `decision=applied` and control reload ok.
8. Later env telemetry uses the same new assignment sha with provider load ok.
9. Survival/eval curves are checked separately; they are not implied by the
   contract proof.

Latest health snapshot:

- Training progress inspector: 18/18 roots and env logs present; 18/18 applied
  refresh index 1; 0/18 have `iteration_10000`; no eval manifests yet.
- Tournament progress: all 153 startup-checkpoint pairs have started, estimated
  3213/3213 games launched, zero known failures, but final shard summaries and
  rating reducer output are not written yet.

## 2026-05-14 22:58 EDT New-Checkpoint Continuation Spawned

Facts:

- Startup round `round-000000` completed: 153 pairs, 3213 games, 0 failures,
  ratings written.
- It reports `stable=false` because the largest Elo movement was about 67. That
  means "complete but still moving," not "failed."
- Four patched trainers now have `iteration_10000` checkpoints.
- Intake status saw latest refs and a 22-ref seen pool.
- `intake-drain` spawned continuation call
  `fc-01KRMS2YQTHQGCRJA10R7PZ84J` over 23 checkpoint refs with
  continue-from-latest true.

Important correction:

- A direct detached `mode=rating --continue-from-latest` command was also
  spawned once, but it had no checkpoint refs and estimated a zero-size plan.
  Treat it as a harmless no-op, not as proof.

Next action:

- Monitor `round-000001` for the live rating run.
- If it completes cleanly, use the promotion controller path. Do not manually
  assemble leaderboard/assignment/pointer proof by loose commands unless the
  controller is unavailable and every artifact check is recorded.

Controller readiness update:

- `scripts/promote_curvytron_rating_round.py` now has the control-volume path
  needed for this live lane:
  - `--assignment-target-volume control`;
  - `--mirror-assignment-checkpoints-to-control`;
  - `--refresh-pointer-ref <shared pointer>`;
  - `--refresh-pointer-volume control`.
- Focused tests and ruff passed.
- Use these flags for round-1 promotion so the trainers can see the assignment
  through their existing `/control` refresh path.

## 2026-05-14 23:11 EDT Promotion Done, Waiting for Trainer Refresh

The controller promotion for `round-000001` succeeded:

- Round artifacts agreed.
- Leaderboard snapshot was published with 25 active rows and 0 provisional rows.
- Assignment was written to the control Volume.
- Two selected checkpoint opponents were mirrored into `/control`.
- The shared refresh pointer was updated to the new assignment sha
  `8238b0d92a73192a63709f6cf895f0bab07fa90f93c47d48b32b45639dc9c1fb`.

Next monitor:

- Query env/refresh telemetry for the new sha.
- Required proof is not "pointer uploaded"; it is trainer evidence:
  refresh index 2, `decision=applied`, control reload ok, and later
  `env_steps.jsonl` rows with the new sha and provider load ok.

## 2026-05-14 23:20 EDT Trainer Refresh Index 2 Observed

The env telemetry scan found the promoted assignment inside live self-play:

- wanted assignment sha:
  `8238b0d92a73192a63709f6cf895f0bab07fa90f93c47d48b32b45639dc9c1fb`;
- env logs present: 18/18;
- first full scan rows with wanted sha: 2/18 trainers;
- later 64 MiB tail scan rows with wanted sha: 6/18 trainers;
- later 64 MiB tail scan rows with wanted sha and provider load ok:
  6/18 trainers.

Those trainers had `max_refresh_index=2`, so the chain has been observed at
least once:

1. trainer wrote post-start checkpoint;
2. intake discovered it;
3. tournament rated it;
4. controller published leaderboard snapshot and assignment;
5. shared `/control` pointer changed;
6. live trainers refreshed and used the new assignment.

Remaining checks:

- watch the other active trainers for refresh index 2;
- confirm future checkpoints continue through the same path without manual
  rescue;
- evaluate survival/score movement separately from plumbing health.

Monitoring note:

- `artifacts/local/curvy_v2refresh18p_env_tail_inspector.py` now provides a
  cheaper repeated check over recent env rows instead of rereading each full
  `env_steps.jsonl` file.

## 2026-05-14 23:24 EDT Training Jobs Still Running

The 18-run status check against the real v2refresh batch shows:

- 18/18 trainers running;
- 18/18 have checkpoint files;
- 18/18 have eval artifacts;
- latest checkpoints are between `iteration_10000` and `iteration_30000`.

This means the live-loop monitor should continue instead of relaunching yet.
Learning quality remains open: some rows have useful-looking survival means, but
several are collapsed or near-collapsed on one action.

## 2026-05-14 23:26 EDT Learning Signal Is Mixed

Eval-summary trend:

- 9/18 latest eval means are above their first eval mean;
- about 12/18 have at least one improved checkpoint;
- several rows are action-collapsed or near-collapsed.

Treat this as early evidence that the batch is producing useful data, not as a
claim that the training recipe is solved.

## 2026-05-14 23:27 EDT Round-2 Promotion

Repeated closure reached the tournament again:

- `round-000002` completed with 37 checkpoints, 666 pairs, 13,986 games, and
  zero failures.
- The published leaderboard had 37 active rows and 0 provisional rows.
- Round 2 was promoted into `/control`.

Sha clarification:

- The refresh pointer uses the canonical assignment JSON sha.
- The raw file-byte sha can differ because the assignment file is pretty-printed
  on disk. That raw file sha is not what the trainer compares.
- A false-alarm patch to use raw file sha was reverted after checking the trainer
  resolver.
- Focused tests remain clean: 8 passed, ruff clean.
- The live pointer is set to the canonical round-2 assignment sha:
  `f6e60a2d7052ef7017b6a16dec62b324492cec0e0170389b678334f7493a5c34`.

Next monitor:

- wait for active trainers to pick up refresh index 3 with sha
  `f6e60a2d7052ef7017b6a16dec62b324492cec0e0170389b678334f7493a5c34`.

## 2026-05-14 23:31 EDT Round-2 Assignment Used In Env Rows

The round-2 assignment has now appeared in trainer env telemetry:

- 4/18 trainers have recent env rows with round-2 sha
  `f6e60a2d7052ef7017b6a16dec62b324492cec0e0170389b678334f7493a5c34`;
- all 4 have provider-ok rows;
- the observed refresh index is still 2, so this is not yet the clean
  r1-then-r2 index-3 proof for every row.

Keep monitoring after a sleep. Do not republish just to force a counter; the
important fact is that the second tournament-derived assignment is being used.

## 2026-05-14 23:37 EDT Repeated Loop Proven

After a 5-minute sleep:

- 12/18 trainers have recent env rows with the round-2 assignment sha;
- all 12 have provider-ok rows;
- 6 trainers show refresh index 3.

This is the clean repeated-loop proof. We have observed a later checkpoint set
flow into the tournament, a second leaderboard-derived assignment flow back out,
and running trainers consume it after already having applied an earlier
assignment.

## 2026-05-14 23:39 EDT Survival Trend Improving But Uneven

The later eval-summary shows:

- all 18 trainers still running;
- checkpoints now up to `iteration_40000`;
- 11/18 latest eval means above their first eval mean;
- about 14/18 with at least one improved checkpoint;
- several rows still collapsed or near-collapsed.

So the run is worth leaving alive and monitoring. It is not yet evidence that
the recipe is solved.

## 2026-05-15 00:20 EDT Deep Monitor Summary

What is working:

- Trainers are still producing checkpoints, now roughly `40k` to `80k`.
- Intake/subscriber sees the new checkpoints; the seen pool is over 100 refs.
- Rating continues: `round-000003` completed with 41 active checkpoints, 820
  pairs, 17,220 games, zero failures.
- Public/control has now been advanced to round 3 by the controller:
  `v2refresh18p-live-r3-20260515a` ->
  `v2refresh18p-live-r3-assignment-20260515a`.

What is not fully automatic:

- Before the controller run, public leaderboard and `/control` were still on
  round 2 while rating had already completed round 3.
- The batch therefore has automatic training -> intake -> rating, but promotion
  from completed rating to public leaderboard / `/control` is still an explicit
  Coach/controller action.

Learning read:

- Survival is only weakly up in collector metrics:
  `145.70 -> 146.09` average latest, best-seen average `151.04`.
- Checkpoint eval tables show bigger noisy spikes, but not a stable upward
  curve.
- Clean rows beat stochastic `so10rep10` rows in the current survival audit.
- `blank10` mixed-rank slots look best on average.
- Policies are changing and recent env action distributions are not collapsed,
  but this is not yet strong learning.

## 2026-05-15 00:25 EDT v2refresh18p Monitor

Current evidence:

- 18/18 patched refresh trainers are still running.
- All rows have fresh-ish checkpoints/evals/GIFs. Latest checkpoints range from
  iteration 40k to 80k.
- The shared `/control` refresh pointer points to
  `v2refresh18p-live-r3-assignment-20260515a`, sha
  `fc88410cab12dbde2258cc23213aaf53194605a14b05b9f20c1d23635812f8fb`.
- Recent env tail inspection shows round 3 has been consumed by 5/18 rows.
  The other rows still show the round-2 assignment sha in their recent tails.
- Latest completed tournament snapshot is round 3 with 41 rated checkpoints.
- A large round 4 was spawned after draining intake: 6,670 pairs / 140,070
  games. Its progress artifact is running at `game_map_started` with zero
  completed games so far.

Learning read:

- Average eval survival is up from about 131.3 first-eval mean to about 148.7
  latest-eval mean.
- 13/18 rows are up on latest eval versus first eval.
- 17/18 rows have at least one best eval above their first eval.
- This is positive but noisy: many rows peak and then regress, so this is not a
  clean monotonic improvement claim.

Action/collapse read:

- Direct train action observability remains absent in the status table.
- Latest eval marks one row collapsed:
  `survbonusout blank20 so10rep10`, top-action fraction `0.989`.
- Several GIF summaries are also very action-heavy, so collapse monitoring
  remains active.

Next checks:

- Re-run the env tail inspector for sha
  `fc88410cab12dbde2258cc23213aaf53194605a14b05b9f20c1d23635812f8fb`
  after another refresh interval; pass target is 18/18 rows showing the sha
  with provider-ok evidence.
- Check round 4 progress again. Do not promote it until reducer artifacts exist,
  pair/game counts agree, and failures are zero.
- Use explicit run ids from
  `artifacts/local/curvytron_tonight18_manifests/curvy-v2refresh18p-20260514b/submission-full-control.json`;
  the run-status helper does not currently have a `v2refresh18p` preset.

## 2026-05-15 00:37 EDT Round-3 Refresh Follow-Up

The round-3 assignment has now clearly come back into the running trainers:

- event-level proof: 17/18 rows have an explicit `decision=applied` refresh
  event for round-3 sha
  `fc88410cab12dbde2258cc23213aaf53194605a14b05b9f20c1d23635812f8fb`;
- each applied event reports `env_ready_report.ok=true`, `env_num=256`, and
  `volume_reload.ok=true`;
- env-tail proof: 16/18 rows have recent env rows using the round-3 sha with
  provider-ok evidence.

This is enough to say the round-3 loop is working for almost all rows. Continue
monitoring the laggards, but the main new concern is not trainer refresh; it is
round 4 scheduling.

Round 4 status correction:

- The progress file is not a reliable live completion counter while the parent
  Modal map is blocked on game work. It remains at `game_map_started` with zero
  completed pairs even though logs show successful `r000004` game workers.
- The round-4 input appears to have been rewritten during monitoring. Latest
  input read showed 1,176 pairs / 24,696 games from 49 checkpoint players.
- This points to overlapping continuation parents or repeated intake drains
  touching the same round id. It does not mean games are dead.
- Do not promote or use round 4 until final reducer artifacts exist and the
  round/input/roster hashes agree.
