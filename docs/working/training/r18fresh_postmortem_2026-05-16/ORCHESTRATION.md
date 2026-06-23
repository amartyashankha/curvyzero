# Orchestration

Last updated: 2026-05-17 11:25 EDT.

This is the tactical board for the r18fresh postmortem and next-batch setup.
Unlike `OPERATING_PATTERNS.md`, this file is allowed to change frequently.
Use `TASK_BOARD.md` for the compact lane table.
For general feedback-loop architecture, observability, and policy-observation
contracts, start in `docs/working/training/curvytron_feedback_loop`.

## Current Objective

Current objective is cz26 live-loop validation and cleanup. The 18-run
`r18fresh` analysis remains useful background, but the active work is proving:
trainers write checkpoints, intake sees them, tournament game batches rate
them, trainer-facing leaderboard/export advances, and running trainers consume
those refreshed opponents.

The current phase is post-launch validation plus guardrails. For the live
`cz26-full-20260517a` batch, answer from tooling/artifacts rather than memory:

1. Which checkpoints were written?
2. Which checkpoints reached intake?
3. Which checkpoints were scheduled and played by the tournament?
4. Which rows became active/top-ranked in the tournament?
5. Which trainer-facing export contained them?
6. Which trainers applied that export and loaded those exact opponents?
7. Which survival/reward/action metrics moved after that point?

Current live fact at 07:10 EDT: latest known published rating is still
`round-000015` / `919`, but the tournament is actively working on internal
game-batch artifact `round-000033`. It contains `2192` checkpoints, `1075`
pairs, and `22575` games. This covers the current intake pool and is current
adaptive coverage, not all-pairs. Do not call the full loop validated until a
rating beyond `919` publishes, trainer refresh uses it, and `trainer-proof`
sees trainers load that newer export.

07:15 EDT recheck: `round-000033` is producing output. The status probe saw
`21` completed game summaries with newest output about `11s` old, and the
Modal call summary saw `44` tournament game calls succeeded and `52` pending.
Latest rating remains `919`. Intake has advanced to `2252`, so `60`
checkpoints arrived after this active batch started; that is normal backlog for
the next batch, not proof of a stall.

07:28 EDT recheck: `round-000033` remains active and not published. Latest
rating remains `919`; intake is `2308`, queue length `116`. Broad probe still
sees `21` completed game summaries, newest about `203s` old. Compact call graph
sample shows `45` tournament game successes and `51` pending. There are no hard
blockers in status; wait for remaining games/reduce, then refresh/prove trainer
consumption only after rating advances.

07:44 EDT recheck: `round-000033` remains active and is still producing recent
output; newest probed output is about `19s` old. Latest rating is still `919`.
Trainer refresh generation moved to `8`, but its source is still
`round-000015`, so this is an old-rating export refresh, not new loop proof.

08:15 EDT recheck: `round-000033` remains active and not published. The sampled
Modal call graph improved to `67` tournament game successes and `29` pending,
so progress is real but slow. The newest probed output age is about `599s`,
near the stale threshold, so the next wait should be short. If output goes
stale or recovery skips, debug progress/reduce; do not duplicate the active
drain.

08:20 EDT recheck: the narrow output probe is stale, but sampled Modal game
calls still progressed from `67` to `75` successes and pending fell from `29`
to `21`. Treat this as slow active games rather than a stuck batch unless
call-graph progress also stops.

08:26 EDT recheck: sampled Modal call graph now shows all `96` tournament game
children as `SUCCESS`; rating loop and rating round are still pending. The next
gate is rating-round reduction/publish, not game execution.

08:31 EDT log check: app logs still show fresh `round-000033` game completions
and runner disappearance/reschedule messages. The remaining issue is slow game
execution/retries before reduce can happen. Do not duplicate the active drain.

08:39 EDT recheck: the active batch is still `round-000033`; latest rating is
still `round-000015` / `919`; trainer export generation `10` still sources that
old rating. Intake has advanced to `2668` and queue length is `476`, while the
active batch covers the `2192` checkpoints present when it started. App logs
show fresh completions at `12:36-12:39Z`, so the batch is alive. Do not read
`probe_completed_game_count=21` as total progress: it is only a narrow liveness
sample. Do not read the capped Modal call graph as all game calls either. The
next main-thread action is to make progress/status tooling say this plainly,
then continue monitoring until rating publish or current-code recovery.

08:52 EDT proof: `round-000033` published and became latest rating with `2192`
checkpoints and `22575` completed games. `refresh-if-ready` wrote generation
`11`, snapshot `auto-r000033-g11-50405322`, sourced from `round-000033`, with
`100` active rows and `24` rewritten pointers. `trainer-proof` then found
generation-11 targets already consumed by trainers: `3/136` runs have a target
as latest-applied; provider rows are `4548 ok`, `0 false`, `2115 null/pending`.
The feedback path has been observed once. Remaining live work: the `522` newer
queued checkpoints need the next bounded drain, and trainer catch-up should be
rechecked after refresh cadence time.

08:56 EDT follow-up: spawned exactly one next bounded drain
`fc-01KRV02A38VYDJPE3JWM4A0W4M`. It returned and spawned rating loop
`fc-01KRV02Z7H8Q7466J50EPSPZ7E` for the `2714`-checkpoint desired pool. Queue is
now `0`, but no new active game-batch artifact is visible yet. Logs say the
rating loop is waiting for CPU worker capacity. Do not spawn another drain while
that rating loop call is pending.

09:08 EDT reorientation: the earlier pending rating loop was part of a crowded
app state. The app was stopped/redeployed, and one fresh drain
`fc-01KRV0JK89X9FFFYHHCZEMFNBJ` spawned rating loop
`fc-01KRV0JQYM98FDTZD6Q55VY04B`. That loop created active internal game-batch
artifact `round-000034`, with `2739` checkpoints, `1348` pairs, and `28308`
games. A liveness probe saw fresh game output. Latest rating remains
`round-000033` / `2192` until this active batch reduces and writes ratings.
Current intake is `2805`, so `66` checkpoints arrived after the active batch was
created. Do not call the system fully stable yet: the truthful state is one
complete feedback pass plus the next pass actively running.

09:25 EDT check: the real launch manifest is `136` rows (`96` Grid A, `40`
Grid B), all submitted as `spawned`, all with the pinned old r18fresh rank-1
initial checkpoint. Grid B includes pure slot controls `b100`, `w100`, and
`r1` as normal rows, not a separate third grid. Fast trainer status shows
`136/136` heartbeats and progress files, `134` running, `2` failed:
`cz26a-r013-out67-n0-imm0-b20w05r1` and
`cz26b-r028-out50-n10-imm10-b20w05r1`. Trainer-side checkpoint artifacts total
`2910`, with per-run counts `8..36` and progress iterations `30000..220000`.
Tournament intake is `2877`; active `round-000034` covers `2739`, so the
tournament is behind the newest trainer checkpoint writes but actively playing
the current batch. Latest rating remains `round-000033` / `2192`.
The separate `cz26c` E2E canary is completed with `350` checkpoints and latest
progress iteration `17385`.

09:47 EDT follow-up: after a 15-minute wait, active `round-000034` is still
running and latest rating remains `round-000033` / `2192`. Intake is `2989`,
queue is `184`, and `250` checkpoints arrived after `round-000034` was created.
Activity-enabled status saw fresh game output about `128s` old. Trainer-facing
export is generation `12`, still sourced from `round-000033`. `trainer-proof`
for generation `12` scanned all `136` runs and found `62/136` with a target
assignment as latest-applied, `48666` provider-ok rows, and `0` provider-false
rows. The current blocker is not trainer consumption; it is waiting for
`round-000034` to publish beyond `2192`.

10:19 EDT follow-up: after another 30-minute wait, active `round-000034` is
still running. Latest rating remains `round-000033` / `2192`. Active batch age
is about `60m`; latest sampled game output age is about `341s`, still below the
`600s` stale line. Intake is now `3136`, queue is `331`, and `397` checkpoints
arrived after the active batch was created. Trainer-facing export generation
advanced to `13` but still sources `round-000033`, so this is not new tournament
rating progress. Continue waiting; no duplicate drain.

10:30 EDT reorientation: main launch is still alive and producing checkpoints.
Fast trainer status saw `136/136` heartbeats/progress files, `134` running,
`2` failed, `3185` checkpoint artifacts, and max iteration `250000`.
Tournament intake is `3193`; latest published rating is still `round-000033` /
`2192`; active `round-000034` covers `2739` checkpoints, `1348` pairs, and
`28308` games. The cheap liveness probe briefly reported stale output, but app
logs show successful `round-000034` games at 10:29 EDT and a wider liveness
probe saw output about `44s` old. `trainer-proof` is strong for the current
generation-13 export: `102/136` latest-applied target count, `74170` provider-ok
rows, `0` provider-false rows. The expensive total progress probe timed out at
`300s`, so the operator truth is: games are landing, but total active-batch
completion count is still not cheap to observe. Next proof remains
`round-000034` publish, refresh from it, and trainer-proof from that newer
rating.

11:20 EDT fix/recovery update: the control-plane diagnosis was real. Stale
active game-batches were blocked before recovery could scan them, and adaptive
pairing could inflate `pairs_per_round=300` into `1348` pairs. Patched and
deployed: stale active batches can enter recovery, scheduled drain tick probes
activity, adaptive selection treats `pairs_per_round` as a hard cap, large
recovery scans use bounded activity probing, and operator output includes the
skip decision. Focused tests passed (`19 passed`, then `18 passed`). Live
recovery scan for `round-000034` returned `not_skippable`: bounded scan saw
`5103 / 28308` completed games, `243 / 1348` started pairs, and fresh activity.
The batch is still slow and too large, but not dead. Future batches should be
bounded to `300` pairs.

11:25 EDT update: wrote
`WHAT_THE_HELL_IS_GOING_ON_HANDOFF_2026-05-17.md` as the current plain-language
handoff. A fresh status check still shows latest published rating
`round-000033` / `2192`, intake `3427`, queue `622`, and active
`round-000034` at `4414 / 28308` root completed games. The cheap liveness
sample is stale, so a patched `drain-if-ready` recovery scan is running now.
Do not spawn a duplicate drain while that scan is in flight.

11:26 EDT update: recovery/drain found `round-000034` complete and published it
as latest rating (`2739` checkpoints, `28308 / 28308` games). It spawned the
next rating call `fc-01KRV8JHYV02SJ8GNH1SJPGQW4` for the `3427` checkpoint
pool. The trainer-facing export is still stale from `round-000033`, so the
main thread is running `refresh-if-ready` and status in parallel. Next proof is
trainer-proof against a `round-000034` export.

11:31 EDT update: `refresh-if-ready` wrote generation `16` from `round-000034`.
The next active `round-000035` is bounded (`300` pairs / `6300` games), but its
input is exactly the old latest `2739`-checkpoint pool and excludes the `688`
newer refs in intake. Patched control/skip logic so this old-pool active batch
can enter recovery and be skipped as `different_spec_already_rated_pool`.
Focused tests passed and deploy is running. Next action after deploy: run
`drain-if-ready`, then confirm the next active batch includes newer refs.

## Current Runtime Assets

See `ASSET_REGISTRY.md` for the live/archived/cleanup status. Do not trust
memory for app, Volume, Dict, Queue, arena, rating, or run-prefix names.

## Current Naming Contract

Use `NAMING_CONVENTIONS.md` before generating any new manifest. The current
batch names are `cz26a` for Grid A, `cz26b` for Grid B, and `cz26c` for
canaries. New visible run names should carry only the row and active axes, for
example `cz26a-r017-out33-n10-imm0-b20w05r1`. Exact slot counts, seed
checkpoint refs, refresh cadence, arena ids, and volume names belong in
structured manifest fields.

## Delegated Lanes Completed

- Training analysis: matched eval grid, latest/best caveats, checkpoint-rate
  unevenness, and per-setting survival readout.
- Tournament analysis: round 32/33 status, top-band composition, mid-run versus
  latest pattern, and current rank-1 policy.
- Top-100/top-10 explanation: website shows raw rating rows; active status is
  the top-100 trainer-facing filter.
- Testing audit: local coverage exists for most components, but no single local
  synthetic test closes the whole loop.
- Data-source audit: current operator truth should come from deployed
  status/control tooling, not old prose or dashboards. Modal Volume artifacts
  are backing evidence for debugging.
- Existing tooling lane: `analyze_curvytron_eval_curves.py` works on cleaned
  eval-status JSON for `mean_survival` and `mean_training_reward`.
- Matched-comparison lane: manifest axes confirm no batch-size correction is
  needed inside the completed r18fresh analysis; all rows in that historical
  batch shared batch size, collector count, simulation count, H100 compute lane,
  and eval seed count. This is not current launch guidance.
- Tournament lane: active cap is real at 100 rows, raw website rows can exceed
  that, and tournament game duration rose from `131.16` to `162.20` physical
  steps over rounds `0..35`.
- Perspective lane: training uses `random_per_episode` learner seats on the
  current source-state path; tournament eval uses balanced randomized seating.
- Metadata lane: checkpoint loading now requires explicit policy observation
  surface/backend metadata and rejects hidden fallback for actual eval loads.
- Observability critique lane: the missing artifact is a stitched lineage table
  across learner update, checkpoint, intake, tournament row, export generation,
  assignment SHA, and provider-load rows.
- Optimizer speed lane: current broad training default is L4/C256/N256/batch64
  with sim8 and `browser_lines + simple_symbols + cpu_oracle`. Fresh profile:
  best L4 `713.83` env steps/s versus best H100 `1001.94`; L4 throughput is
  about `28.8%` lower, acceptable for cheaper broad runs. H100 is now an
  explicit expensive/sentinel override, not the default.

## Delegated Lanes In Flight

- Tool-inventory lane: find existing commands before proposing new helpers.
- Own-latest control lane: find/analyze the selected control rows and compare
  against the original clean no-outcome rows.
- Instrumentation hook critique: identify smallest code-level emit points for
  the lineage table.
- Retrospective signal critique: identify which past-batch questions still
  cannot be answered cleanly and convert them into next-batch acceptance
  criteria.

## Main-Thread Duties

- Keep this doc and `TODO.md` current.
- Merge subagent findings into `FINDINGS.md`.
- Run focused local tests after doc cleanup and any code changes.
- Extract or pin the top-10 current checkpoint refs before any destructive
  cleanup.
- Decide the next experiment shape after the analysis is documented.
- Keep the main thread as the integrator. Delegate bounded critiques, then fold
  results into `SIGNALS.md`, `TODO.md`, and `FINDINGS.md`; do not scatter
  stale one-off docs.

## Immediate Next Actions

1. Use the deployed-function script for live operations, not `modal run`:
   `uv run --extra modal python scripts/curvytron_live_loop_control.py --action status --activity-probe-pairs 4 --lookahead-batches 64`.
   `modal run` creates a temporary scheduled app and has already caused a
   stopped-app `ConflictError`.
   The script has a local operator lease for drain requests, so repeated
   operator calls do not blindly create duplicate drains while Modal is still
   scheduling the previous one. The default pending-call probe is now compact;
   use `--full-drain-call-graph` only for deep Modal call-graph debugging.
   Compact status now includes `proof_chain` and `blockers`; use those first.
2. Current live state: active game-batch artifact `round-000034` is running.
   Latest rating is still `round-000033` / `2192`; active batch covers `2739`
   checkpoints and current intake is `2805`. Fresh game output exists. Wait for
   this batch to reduce and publish. Do not stack duplicate drains.
3. If game output appears, keep waiting for reduction. If recovery later skips
   it, the skip record must include current-code output-scan fields. If no game
   output appears after a reasonable wait, inspect scheduler/game call status
   before taking any destructive cleanup action.
4. Current trainer-refresh state: generation `7` was written from
   `round-000015`, and `24` assignment pointers were rewritten. This is still
   not a newer tournament rating.
5. Current trainer-consumption state: earlier `trainer-proof` scanned `136` runs and
   found `48` with a generation-4 target assignment SHA as latest applied,
   `43723` target provider-ok rows, and `0` target provider-false rows.
6. After `round-000033` published, `refresh-if-ready` was run and wrote
   generation `11` from `round-000033`.
7. `trainer-proof` has proven partial generation-11 consumption. Re-run after
   trainer refresh cadence time and expect latest-applied target count to rise.
8. Fast trainer status command:
   `modal run src/curvyzero/infra/modal/lightzero_curvytron_run_status.py --run-ids <manifest-run-ids> --output fast-summary --chunk-size 16 --chunk-workers 8`.
   The older full table can time out on all `136` rows because it scans heavy
   eval/GIF/history fields.
9. Keep `CZ26_CURRENT_TRUTH_2026-05-17.md`, `TASK_BOARD.md`, and
   `TESTING_AND_GAPS.md` synchronized after each real finding.
10. Continue stale-doc cleanup so old r18fresh/prelaunch language is clearly
   historical.

## Open Follow-Ups

- Confirm whether `round-000033` publishes a rating beyond `919`; if it does
  not, inspect its skip/reduce decision with the deployed status fields before
  spawning more work.
- Decide top seed policy: raw active top 10, best-per-run top 10, or best-per
  setting top 10.
- Preserve `ownlatest` control results separately from tournament-attached runs.
- Add an operator health strip to the tournament site using the same fields as
  `loop-status` / `loop-control`, so "current arena", "latest rating batch",
  "active rows", "newest checkpoint", and "trainer export age" are visible
  without manual spelunking.
- Recheck Modal GPU prices before making a cost claim; keep the decision as a
  ratio until live prices are confirmed.
