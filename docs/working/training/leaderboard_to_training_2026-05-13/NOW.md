# NOW: Coach / Tournament Control Panel

Last organized: 2026-05-16.

Read this first. This page is the current plain truth, not a full history.

For the cleaned current architecture, read
`CURRENT_TOURNAMENT_PIPELINE_2026-05-16.md` first. Older "proof lane" entries
below are historical evidence unless that file cites them as current.

## Immediate Plain Answer

- 2026-05-16 cleanup pass: `README.md` now points first to
  `CURRENT_TOURNAMENT_PIPELINE_2026-05-16.md`, which is the current plain
  architecture. Tournament CLI current-lane modes now default to the shared
  current arena/rating ids instead of inventing a random arena when
  `--tournament-id` is omitted, and `--mode current` prints the static
  automation contract without spawning work. Intake scan cadence now lives in
  the shared CurvyTron contract.
- 2026-05-16 11:35 EDT source-of-truth correction: the tournament should be
  the source of truth for "best policy so far", but that does not mean the
  current best policy is impressive. The active tournament latest is now
  `round-000029` with `549` rated rows, `531` nonzero rows, max iteration
  `310893`, one-frame eval, balanced random seating, `21` games per pair,
  `max_steps=1048576`, and eval policy mode. The trainer-facing Modal Dict
  export is one handoff layer behind at generation `18`, snapshot
  `auto-r000027-g18-f8a118b4`, published at
  `2026-05-16T15:09:59Z`. Plain meaning: the tournament truth store is alive
  and ahead of the trainer export; the quality issue is that the top policies
  are still modest mid-run checkpoints, not a clean breakthrough.
- 2026-05-16 11:40 EDT manual controller handoff proof: a
  `training-candidate-auto-refresh` call against the active arena refreshed
  from `round-000029` with `549` rows, `100` active rows, generation `20`, and
  rewrote all three recipe pointers. Immediately after, the trainer-facing
  Modal Dict pointer read as generation `21`, snapshot
  `auto-r000029-g21-65ecaffa`, published at `2026-05-16T15:39:55Z`, with top
  checkpoint IDs from the same active tournament. Plain meaning: the
  tournament truth store is feeding the trainer-facing export layer.
- 2026-05-16 11:47 EDT GIF setting truth: the active ranking arena was seeded
  with persisted no-GIF settings, so completed tournament rounds are not
  capturing tournament GIF samples. Its intake config and `round-000029/input.json`
  both have `save_gif=false`. It does have
  `gif_sample_games_per_pair=5` and `gif_fps=800.0`, but those settings do
  nothing while `save_gif` is false. Source defaults did not regress:
  `DEFAULT_SAVE_GIF=True` and `DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR=5` are still
  in code. The current intake config was corrected for future rating work; old
  completed no-GIF rounds remain no-GIF unless rerun. For any future visual
  arena, verify persisted intake/rating artifacts, not just source defaults.
- 2026-05-16 11:43 EDT own-latest selected3 proof: fresh lane
  `curvy-ownlatest-staticmix-20260516b` is not part of the active tournament
  watch by default because the watch is scoped to the 18 `r18fresh` run IDs.
  Discovery found latest refs for all three rows: two are still
  `iteration_0`, and row `r011` reached `iteration_10000`. Row `r011`
  applied its run-local own-checkpoint assignment at train iter `10789`
  (`sha=23ebfe2d...`), and env-tail proof found `19,240/19,240` target rows
  loading that frozen `iteration_10000` checkpoint with
  `opponent_provider_load_ok=true` and `0` provider-load false rows. This
  proves the own-latest control mechanism on one row after the 11:23 deploy.
  Do not inject its iteration-0 refs into the tournament; exact-submit nonzero
  refs later if they become useful.
- Portfolio framing: the 18 trainers do not all need to improve. This setup is
  useful if a few runs discover strong checkpoints and the tournament preserves
  those checkpoints as training opponents. Current `r18fresh` evidence says
  every row found a better checkpoint than iteration 0, but many latest
  checkpoints regressed. The next learning question is therefore "did any
  tournament-ranked checkpoint become meaningfully good enough to steer the
  batch?", not "did every trainer curve go up monotonically."
- 2026-05-16 survival-quality investigation is active. Corrected current read:
  best survival improved in `18/18` r18fresh rows, but latest survival improved
  in only `10/18`; mean survival is first `159.9`, best `246.0`, latest
  `175.4`; only `4/18` latest checkpoints are within 10% of their own best.
  Historical action-collapse flags appear in `13/18` ladders, but latest evals
  are not collapsed by the 0.95 top-action threshold. Plain meaning: this batch
  often discovers better mid-run checkpoints and then regresses. See
  `survival_stagnation_investigation_2026-05-16.md`. Do not use older survival
  snippets in this doc as the current conclusion.
- 2026-05-16 11:02 EDT fresh read: the live tournament is still advancing. The
  active arena now has `round-000026`, `530` rated rows, `512`
  nonzero-iteration rows, max checkpoint iteration `310893`, `300` pairs, and
  status split `97` active / `24` provisional / `409` retired. It is still not
  a final-stable rating snapshot (`stable=false`, `max_abs_delta=22.31`), but
  ingestion/rating is moving.
- Latest parallel survival deep-dive: `r18fresh` remains "learns, then
  regresses": `18/18` rows improved at best, `12/18` improved at latest, mean
  first/best/latest survival `160.2 / 247.0 / 181.6`, only `2/18` latest rows
  within 90% of own best, and `0/18` latest action ladders collapsed. The
  selected no-feedback `r18nofb` control is still early: `3` rows, `5` eval
  points in the subagent read, mean first/best/latest `156.4 / 160.4 / 151.5`,
  refresh events/applied both `0`.
- Trainer app redeployed at 2026-05-16 11:03 EDT with the local hardening:
  atomic JSON writes, diagnostic launch knobs, and own-checkpoint moving-control
  support. Existing already-running jobs keep their old image; newly submitted
  trainer jobs use the updated deployed app.
- Trainer app redeployed again at 2026-05-16 11:23 EDT with two diagnostic
  fixes: passive learner metrics (`learner_metrics.jsonl` plus
  `learner_metrics_latest.json`, surfaced by run-status) and an own-latest
  refresh fix. For own-latest controls, the trainer now reads the run-local
  assignment pointer/checkpoint directly instead of calling `Volume.reload()` on
  the same runs volume while TensorBoard may have files open. Normal tournament
  / control-plane refreshes still reload external state. Local verification:
  `156 passed, 3 skipped`; ruff clean.
- Parallel docs-only reviews completed:
  `modal_debugging_patterns_2026-05-16.md`,
  `original_vs_current_deep_compare_2026-05-16.md`, and
  `control_run_status_2026-05-16.md`. Control status says the static
  no-feedback selected3 is a valid no-tournament control with refresh
  counts exactly zero and mixed/slightly-up early survival
  (`155.083 / 166.458 / 158.083` first/best/latest). The old own-latest
  selected3 is correctly shaped but not proof: it is still at iteration 0 and
  hit the runs-volume reload trap that the 11:23 deploy fixes for fresh jobs.
  The stale pre-fix own-latest selected3 train/poller calls were cancelled
  after the fresh `20260516b` lane launched, so they no longer burn compute.
- Small own-checkpoint moving control launched after that deploy:
  `curvy-ownlatest-staticmix-20260516a`, selected rows `r007`, `r009`, `r011`.
  Launch record:
  `artifacts/local/curvytron_tonight18_manifests/curvy-ownlatest-staticmix-20260516a/selected3.launch.json`.
  This is not tournament feedback and not true current-policy self-play; it is
  "learner versus refreshed frozen own previous checkpoint." First proof gate:
  startup/iteration-0; second proof gate: after a nonzero checkpoint, the
  trainer writes a run-local `runs:` refresh pointer and later applies it.
  Initial startup poll: all `3/3` train jobs and pollers are running; two rows
  already wrote `progress_iteration=0`; no refresh has applied yet, which is
  expected before the first nonzero checkpoint.
- Fresh own-latest selected3 launched after the 11:23 EDT reload/metrics deploy:
  `curvy-ownlatest-staticmix-20260516b`, selected rows `r007`, `r009`, `r011`.
  Launch record:
  `artifacts/local/curvytron_tonight18_manifests/curvy-ownlatest-staticmix-20260516b/curvy-ownlatest-staticmix-20260516b.selected3.launch.json`.
  Train calls:
  `fc-01KRRP92QZBVWY8AWP9F65A7HD`,
  `fc-01KRRP92ZXNV5EZABBC30HE81E`,
  `fc-01KRRP936TR1GQD6XDBQSM0FA4`.
  Pollers:
  `fc-01KRRP92MNBMTH8S4WP7012XYC`,
  `fc-01KRRP92V5NVN3MHX56EW8SXVQ`,
  `fc-01KRRP933F2019565RW35WH4FC`.
  Startup poll immediately after launch: pollers running, at least one train
  heartbeat running, no checkpoints/progress/learner metrics yet. Next gate is
  first nonzero checkpoint (`iteration_10000`), then own pointer write,
  refresh apply, provider-ok env rows, and learner metrics visible in status.
- Local code now also includes the own-checkpoint moving
  control producer: after a nonzero exact `iteration_N.pth.tar`, the trainer can
  write a run-local immutable opponent assignment and update a run-local
  `runs:` refresh pointer. The manifest builder can emit this via
  `--own-checkpoint-opponent-refresh` without shared control pointers. Local
  verification: own-refresh smoke manifest built correctly; focused slice
  `146 passed, 3 skipped`; ruff clean.
- 2026-05-16 12:40 EDT current truth: the corrected live feedback loop is
  proven through two complete 18-run refresh cycles, and the deployed
  30-minute scheduled controller has now fired and fed generation 12 back into
  all still-running trainers. The active arena is
  `curvy-r18fresh-live-bounded-dsf1-20260516b` /
  `elo-r18fresh-live-bounded-dsf1-20260516b`.
- Loop proof already closed for generation 9: corrected tournament
  `round-000002` produced a one-frame `latest.json` with `304` rated rows and
  max checkpoint iteration `220000`; `training-candidate-refresh` wrote
  snapshot `r18fresh-dsf1-r2-training-g9-20260516b`, published the Modal Dict
  pointer, and rewrote the three live control refresh pointers. After the long
  wait, all `18/18` running trainers had env telemetry rows with the gen-9
  assignment shas. Tail proof scanned `177,382` post-refresh env rows:
  `177,382` had one of the gen-9 assignment shas, `89,934` were frozen
  checkpoint opponent rows with `opponent_provider_load_ok=true`, and `0` had
  provider-load false. Plain meaning: tournament-ranked checkpoints came back
  into the same running trainers and were actually loaded as frozen opponents.
- The tournament kept moving after that proof. Generation 10 was published from
  `round-000006` as `r18fresh-dsf1-r6-training-g10-20260516b`; it rewrote all
  three recipe pointers with assignment shas `ffc5bc3f...`, `c25af087...`, and
  `673381d4...`. After a 30-minute sleep, all `18/18` running trainers had env
  telemetry rows with the gen-10 shas. Tail proof scanned `177,203`
  post-refresh env rows: `177,203` had one of the gen-10 assignment shas,
  `93,427` were frozen checkpoint opponent rows with
  `opponent_provider_load_ok=true`, and `0` had provider-load false.
- Automation update: `curvytron_training_candidate_refresh_tick` is deployed in
  `curvyzero-checkpoint-tournament-v2` on a 30-minute schedule. Manual smoke of
  that same function published generation 11 from `round-000009`; after the
  requested full 30-minute sleep, the scheduled tick published generation 12
  from `round-000011` as `auto-r000011-g12-65387f58`.
- Current tournament `latest.json` points at `round-000011` with `398` rated
  rows, max checkpoint iteration `290000`, explicit
  `rating_spec.decision_source_frames=1`, `300` bounded adaptive pairs, and
  `6,300` games.
- Newer live-state audit supersedes the previous tournament-count sentence:
  the same active arena has advanced to `round-000023` with `511` rated rows,
  `493` nonzero-iteration rows, max checkpoint iteration `310893`, explicit
  `decision_source_frames=1`, `300` bounded adaptive pairs, and `6,300` games.
  Current status split is `72` active, `132` provisional, and `307` retired.
  Treat this as coherent live standings, not final ratings:
  `stable=false`, `max_abs_delta=15.46`.
- Generation-12 active-trainer proof is closed: status showed `15/18` trainers
  still running and `3/18` completed. Rechecking only the still-running trainers
  found `15/15` with gen-12 target-sha env rows, `146,976` target rows,
  `84,578` frozen-checkpoint provider-ok rows, and `0` provider-load false
  rows. The completed rows had already consumed earlier tournament-derived
  generations, but they are not expected to emit new env rows after completion.
- 2026-05-16 05:43 EDT corrected live proof lane:
  `curvy-r18fresh-live-bounded-dsf1-20260516b` /
  `elo-r18fresh-live-bounded-dsf1-20260516b`. This is now the Tournament Arena
  website default/current marker. URL:
  `https://modal-labs-shankha-dev--curvyzero-checkpoint-tournament--93d419.modal.run/?tournament_id=curvy-r18fresh-live-bounded-dsf1-20260516b&rating_run_id=elo-r18fresh-live-bounded-dsf1-20260516b`.
- Why the previous bounded lane was not enough: it completed and wrote
  `latest.json` with `272` rated rows through max iteration `190000`, but its
  `rating_spec.decision_source_frames` was `null`. Actual runtime inferred the
  one-frame cadence, but the training-candidate controller correctly refused to
  publish a snapshot without explicit one-frame metadata.
- Fix deployed: tournament rating normalization and CLI/intake defaults now
  make `decision_source_frames=1` explicit. Focused tests and ruff passed, and
  `curvyzero-checkpoint-tournament-v2` was redeployed after the patch.
- Current survival debugging control: three no-tournament/static-opponent rows
  are running from `curvy-r18nofb-staticmix-20260516a` with assignment refresh
  disabled. Fresh status: `3/3` train jobs running and refresh counts exactly
  zero. Row `r007` has latest checkpoint/eval at `iteration_20000` with latest
  eval `138.125` mean steps; row `r009` has `iteration_10000` with latest eval
  `167.625`; row `r011` has reached `iteration_30000` but latest completed
  eval is `iteration_20000` at `171.5`. This is still too early, but the static
  control is already mixed rather than an obvious clean learning win.
- Deployed with current trainer build: hot JSON files now use atomic replace, and
  the tonight18 builder/trainer expose diagnostic knobs for `collector_env_num`,
  `n_episode`, `num_simulations`, `batch_size`, `model_support_cap`, `td_steps`,
  and background eval sizing. Focused local bundle: `107 passed, 3 skipped`;
  ruff clean.
- Corrected proof state: seed found `287` checkpoints across the 18 live runs
  and spawned rating call `fc-01KRR2PG4NN24PH9Q1B0QTB90V`. Persisted
  `round-000000/input.json` has explicit `decision_source_frames=1`, `300`
  adaptive pairs, `6,300` games, `21` games per pair, `games_per_shard=21`,
  `active_pool_limit=100`, and `save_gif=false` for this fast proof lane.
  Final `latest.json` exists and the generation-9 controller/trainer/env proof
  is complete; see the current truth block above.
- 2026-05-16 05:26 EDT previous bounded proof lane:
  `curvy-r18fresh-live-bounded-20260516a` /
  `elo-r18fresh-live-bounded-20260516a`. This was the first bounded proof lane.
  The URL is:
  `https://modal-labs-shankha-dev--curvyzero-checkpoint-tournament--93d419.modal.run/?tournament_id=curvy-r18fresh-live-bounded-20260516a&rating_run_id=elo-r18fresh-live-bounded-20260516a`.
- Why this lane exists: the old live lane intake was unstuck, but its latest
  continuation `round-000013` admitted `261` refs and scheduled unbounded
  all-pairs: `33,930` pairs / `712,530` games. That is too slow for feedback
  proof and was caused by live intake retaining all new/provisional refs while
  using all-pairs defaults.
- Fix deployed: live run-id intake now defaults to bounded adaptive scheduling:
  `pair_selection=adaptive_v0`, `pairs_per_round=300`,
  `active_pool_limit=100`. Exact-ref stress/validation lanes can still use
  all-pairs. Regression tests passed and `curvyzero-checkpoint-tournament-v2`
  was redeployed.
- Bounded proof state: seed found `272` checkpoints across the 18 training run
  ids, max checkpoint iteration `190000`, and spawned rating call
  `fc-01KRR1QZ3J2W627HVDK01NFPDQ`. Persisted `round-000000/input.json` has
  `300` adaptive pairs and `6,300` games, with `21` games per pair,
  `games_per_shard=21`, `save_gif=false` for this fast proof lane. It completed
  later, but it is not publishable because the snapshot lacked explicit
  `decision_source_frames=1`.
- 2026-05-16 04:17 EDT validation lane: a fresh clean all-205 tournament proof
  is running under tournament `curvy-r18fresh-validate-all205-20260516a` and
  rating run `elo-r18fresh-validate-all205-20260516a`, detached app
  `ap-VQZzMzRPLR5ZojFpN1iHbR`, rating call
  `fc-01KRQXFK7WSS1ZEEMAW5GYAHFK`. It accepted exactly `205/205` checkpoint
  refs. The persisted `round-000000/input.json` schedules all pairs:
  `20,910` pairs and `439,110` games at `21` games per pair.
- Current validation proof state: Modal logs show the clean lane is actively
  running successful games with `ok=true`, `max_steps=1048576`, and
  `seat_order.mode=balanced_random` with both swapped and non-swapped games.
  As of the latest artifact poll, progress files still show
  `completed_game_count=0` because this code path writes progress at map start
  and then after the full game map returns. Final proof is still pending until
  `ratings.json`/`latest.json` exist with `205` ranked rows and
  `completed_game_count=439110`.
- Operational note: this validation run used `--no-save-gif` intentionally to
  prove submission/game/ranking behavior without huge GIF I/O. It also used
  `games_per_shard=1`, so it is a valid but slow all-games proof: `439,110`
  individual Modal game tasks.
- 2026-05-16 04:30 EDT validation poll: detached app still alive with `505`
  tasks. Recent logs advanced to pair indices around `1975..2034`; parsed tail
  had `175/175` successful games, `0` errors, `max_steps=1048576`, and balanced
  randomized seating. Durable progress/latest/ratings still have no final
  reduction yet.
- Historical website/current fix: the default was briefly pointed at the
  all-205 validation arena. That has now been superseded: the current/default
  arena is `curvy-r18fresh-live-bounded-dsf1-20260516b` /
  `elo-r18fresh-live-bounded-dsf1-20260516b`, and the Tournament Arena labels
  its live standings as `Live Leaderboard`.
- 2026-05-16 recovery hardening after subagent review: skipped/stale rating
  rounds are now treated as consumed when selecting the next continuation round,
  and an explicit retry of a skipped round returns `status=skipped` instead of
  trying to reduce the stale input again. Legacy shard-tally reduction also
  rebuilds missing `wins_by_checkpoint` from embedded shard games. Targeted
  tests passed (`4 passed`) plus ruff, and `curvyzero-checkpoint-tournament-v2`
  was redeployed.
- Why the user saw no leaderboard rows earlier: the validation arena had
  `205` refs in `round-000000/input.json` but no final
  `ratings.json`/`latest.json` yet. The current bounded arena already shows a
  provisional Live Leaderboard, but final durable `latest.json` is still
  pending.
- 2026-05-16 04:03 EDT answer: no, the current large live lane is not fully
  working/proven. The code has important fixes implemented, tested, and
  redeployed, but the live `curvy-r18fresh-live-20260516a` /
  `elo-r18fresh-live-20260516a` artifact tree is polluted by overlapping
  tournament rounds started by older deploys.
- What is implemented and redeployed: stale no-output skip now requires the
  real stale age floor; root `progress.json`/`latest.json`/pair-history writes
  are monotonic; `continue_from_latest` intake uses one active claim instead of
  a new claim for every growing pool; and intake no longer spawns a reducer for
  an unfinished running round. Focused intake/recovery/pointer tests and ruff
  passed before the latest `curvyzero-checkpoint-tournament-v2` deploy.
- Current durable tournament truth: `latest.json` now points at
  `round-000008`, with `98` rated checkpoints, `4,753` pairs, `99,813` games,
  `58` failed games from round progress, `stable=false`, and
  `max_abs_delta=318.85214603560865`. This is a completed stress snapshot, not
  a clean production source for trainer assignments.
- Current dirty-artifact truth: root `progress.json` still points at
  `round-000010` as `running`, while later round folders exist. `round-000009`
  was wrongly skipped by old code; `round-000010`, `round-000011`, and
  `round-000012` have overlapping running artifacts from old workers. Because
  of that, the current root progress pointer is not trustworthy as a live
  service proof.
- Current intake truth: the live manifest exists with `196` seen checkpoint
  refs and `updated_at=2026-05-16T08:02:52.922179Z`. The Queue API currently
  reports length `0`; the manifest still has `queued_checkpoint_count=196`,
  so treat that manifest bookkeeping as suspect until the next clean run proves
  queue drain behavior.
- Current app truth: deployed v2 trainer, tournament, and GIF browser apps are
  up. Two old detached tournament apps are still in `stopping...` state:
  `ap-ily3OHjnYXnun9616HKYGb` and `ap-svRQc9Y8SyxPu5wbJZ9fTT`. Local process
  check shows no runaway local Modal commands, only the editor `ruff server`.
- The canary loop was proven earlier: checkpoint -> intake/subscriber ->
  tournament -> leaderboard/assignment -> same running trainer refresh -> env
  provider rows. The large 18-run loop has not been cleanly proven after the
  latest tournament fixes.
- Current next decision: do not call this large lane production proof. Either
  start a fresh clean tournament/rating id with the current deployed code, or
  deliberately purge/repair the dirty round artifacts before continuing this
  id. In either case, the proof target is simple: new trainer checkpoints enter
  the tournament, produce a clean `latest.json`, a controller writes immutable
  assignments/control pointers, and the same running trainers apply and use
  those assignments.
- Survival sanity check from the most recent recorded read is mixed, not a
  solved-learning claim: latest mean survival was up in `10/18` runs,
  best-so-far survival was up in `15/18`, and aggregate latest mean moved
  `159.2 -> 164.5` steps (`+5.3`).
- 2026-05-16 00:54 EDT reset: the CurvyZero v2 trainer, tournament, and
  GIF-browser apps were stopped. The CurvyZero v2 volumes/dicts/queue were
  deleted and recreated empty. Old/non-v2 CurvyZero storage was also removed.
- Current clean storage is only:
  `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`,
  `curvyzero-curvytron-control-v2`,
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-opponent-leaderboard-live-v2`, and
  `curvyzero-curvytron-checkpoint-events-v2`.
- 2026-05-16 GIF-site fix: the GIF browser crashed because its Modal image did
  not install `numpy`, even though importing CurvyZero env contracts requires
  it. The image now installs `fastapi` plus `numpy`; the current-batch category
  defaults to `curvy-r18fresh-*`; older rows are archived by default; and the UI
  says `Archive` instead of pretending to delete durable run artifacts. Live
  check passed: root redirects to `category=current`, `/api/summaries?limit=5`
  returns `reload_error=null`, and the page renders current `curvy-r18fresh-*`
  rows.
- 2026-05-16 tournament GIF-speed fix: new tournament GIFs use
  `CURVYTRON_TOURNAMENT_GIF_FPS=800` with `1ms` minimum frame duration, about
  10x faster than the previous 80 fps default. Already-written GIF files keep
  their embedded timing until regenerated by a new rating/GIF pass.
- Historical note for the dirty large lane: fresh launch from regenerated
  all-v2 inputs is running. Apps were redeployed,
  scratch manifest `curvy-r18fresh-allv2-20260516a` was built and dry-run
  checked, then the submitter spawned `18` trainers plus `18` pollers. The
  run-id watcher is active as `curvy-r18fresh-live-20260516a` /
  `elo-r18fresh-live-20260516a`.
- Current live proof: all `18` fresh rows wrote `iteration_0`; intake sees
  `18/18` checkpoint refs and queue length `0`; `round-000000` completed
  `21/21` games with `0` failures; `round-000001` is or was running on the
  `18`-checkpoint pool with `136` pairs and `2856` planned games.
- Current live proof has advanced past startup: at about 01:11 EDT, several
  trainers had reached `iteration_10000` and written second checkpoints. At
  about 01:12 EDT, intake saw `24` refs, including `iteration_10000` refs, with
  queue length `0`.
- Historical earlier proof gap: `round-000003` is still rating an
  18-checkpoint pool. We
  now has a resolved continuation proof: logs show `round-000003` completed at
  about 01:12 EDT, and `round-000005` is running with `27` checkpoints,
  `351` pairs, and game logs containing `iteration_10000` checkpoint ids. We
  still need a completed latest snapshot from the newer pool, then immutable
  assignment materialization, and same-running-trainer refresh.
- Everything below that describes `curvy-r18v2-*`, `curvy-r18scratch-*`,
  `curvy-r18v2-bootstrap-*`, or old `round-000004`/`round-000005` state is
  historical unless explicitly regenerated after the 00:54 reset. It is
  evidence about bugs and mechanics only.
- Correction to the current worldview: `stable=true` is not the gate for
  training. It only means the rating numbers barely moved in the latest Elo
  round. That is useful for final claims, but it is too strict for a live
  training loop. Training needs a clearly labeled current opponent snapshot,
  then immutable assignment files, then automatic trainer refresh.
- The intended loop is automatic: trainers write checkpoints; intake discovers
  them; tournament rates them; a Coach-side controller publishes the current
  usable top snapshot; that controller materializes immutable per-recipe
  assignments; trainers refresh those control pointers every `2000` train
  iterations.
- The current large-batch gap is this controller in the live lane. Canary proofs
  showed the pieces work, and local toy tests now cover the controller boundary:
  rating latest JSON -> training-candidate leaderboard -> immutable
  recipe-preserving assignments -> control refresh pointers -> trainer-visible
  pointer resolution. The remote large 18-run lane still has not consumed a
  fresh controller-produced assignment.
- The canary loop is fully proven end-to-end. The large 18-run loop is not yet
  fully proven end-to-end.
- For the fresh 18-run loop, the front half is working at startup and first
  post-start checkpoint scale: trainers wrote `iteration_0`, some wrote
  `iteration_10000`, live run-id intake saw them, and the tournament is now
  rating a continuation pool that includes those newer checkpoints.
- The back half is still waiting remotely: no automatic controller run has
  rewritten the live large-batch recipe control pointers and then been observed
  in the same running trainers. That is the missing proof. `stable=true` is not
  the live-training gate; it is only the final-ranking quality label.
- This is not a one-checkpoint-per-iteration run. The manifest uses
  `save_ckpt_after_iter=10000`. The tournament is large because 18 trainers have
  produced multiple checkpoints and the current intake watch has `92` refs.
- Fresh check: the durable progress JSON for `round-000004` is stale at
  `2026-05-16T03:02:53Z`, but the detached worker app is still alive and logs
  show successful `round-000004` games around pair `3230`, with balanced random
  seats and `max_steps=1048576`. Treat logs/app state as proof the round is
  still moving; do not treat the stale progress timestamp as completion.
- `latest.json` still points at `round-000003`, `stable=false`, with `57`
  rated checkpoints. No large-batch public leaderboard or training assignment
  has been promoted from this run yet.
- Tail finding: `round-000004` logged through final pair index `4185`, but
  `latest.json` still did not advance. Logs show transient Modal Volume I/O
  errors around battle
  `rate-elo-r18v2-boot-r000004-pair-003925-ckpt-037-train-l-vs-ckpt-003-train-l-4edef7cb97`
  and one `OSError` game record for pair `4117`; later Volume listing shows the
  affected game directories/files exist. A detached reduce recovery was spawned:
  app `ap-5EESnCGGRybj40eNKoZLg4`, function call
  `fc-01KRQDRKNV0PQ9MFBMYGWMHVKH`, `mode=reduce`, `round_index=4`.
- Reduce recovery completed. `round-000004/latest.json` is now durable with
  `92` checkpoints, `4186` rated pairs, `87906` games, `failed_game_count=1`,
  `stable=false`, and `max_abs_delta=401.45`. Do not publish this to trainer
  assignments.
- A same-pool continuation bug was fixed and redeployed: the CLI passed
  `spawn_if_existing`, but drain code only honored `spawn_if_empty`, so same-pool
  continuation could be blocked by a fresh old claim after a finished round.
  Focused test `test_intake_drain_spawn_if_existing_allows_same_pool_continuation`
  passes, and `curvyzero-checkpoint-tournament-v2` was redeployed.
- After the fix, drain spawned `round-000005` via app `ap-CHJuWMB4lb4qfENQqv86N3`,
  rating call `fc-01KRQEPEAAFCC595YRMP0R7MB0`. `round-000005/input.json`
  exists with `92` checkpoints, `active_pool_limit=100`, all-pairs, `4186`
  pairs, `87906` games, and `previous_round_id=round-000004`.
- The top-100 rule is the intended default and the current live round input has
  `active_pool_limit=100`. Mature rows below rank 100 are `retired` and
  excluded from future scheduling/training selection, while new under-tested
  rows can remain `provisional` long enough to receive placement games. The
  current round has `92` refs, so top-100 truncation does not change this round.
- Promotion caution: if a future stable large leaderboard is materialized with
  the generic `stable_slots_v1` assignment and written to all three current
  recipe control pointers, it will collapse the three bootstrap recipe shapes
  into one shared leaderboard-derived assignment. That may be the right next
  phase, but it must be a deliberate choice. Do not silently overwrite all
  recipe pointers while pretending the original recipe axes are preserved.
- Current trainer-side survival readout is mixed rather than a clean learning
  win: across 18 rows, latest eval mean is up `+9.17` steps on average and
  best-so-far eval mean is up `+39.67`; latest improved in `9/18` rows and
  best-so-far improved in `16/18`. By reward group, sparse outcome is strongest
  so far (`+41.46` latest mean), survival+bonus without outcome is roughly flat
  (`+2.23`), and survival+bonus+outcome is down (`-16.19`). Treat this as
  “policies are changing and sometimes improving,” not as stable solved play.

## 2026-05-15 All-V2 Reset

- Current operator decision completed: deleted and recreated the active v2
  storage/control lane, then redeployed the CurvyTron apps against that lane.
  Do not launch new training from the old hybrid namespace.
- Current app names:
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`,
  `curvyzero-checkpoint-tournament-v2`, and
  `curvyzero-curvytron-gif-browser-v2`.
- Current Modal Volumes, all opened with `version=2`:
  `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`.
- Current Modal Dict/Queue objects:
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-checkpoint-events-v2`, and
  `curvyzero-curvytron-opponent-leaderboard-live-v2`.
- Historical proof in old/non-v2 or hybrid storage remains useful only as
  evidence about the mechanics. It is not a valid launch input for the new
  all-v2 lane unless explicitly copied or rematerialized there.
- The shared source of truth is
  `src/curvyzero/contracts/curvytron.py`. If a script or app has a different
  default, treat that as a bug.
- Verified at about 2026-05-15 14:10 EDT: v2 trainer/tournament/GIF apps are
  deployed; the old non-v2 Curvy trainer/tournament deployments are stopped.
- Local checkpoint metadata hardening is now part of the launch contract:
  fresh checkpoints write `iteration_N.pth.tar.metadata.json` beside the weight
  file, and tournament discovery/loading reads that sidecar before stale
  run/attempt metadata or defaults. This prevents a checkpoint from being
  evaluated on an observation surface different from the one it was trained on.
- Latest tournament deploy status: `curvyzero-checkpoint-tournament-v2` was
  redeployed after the metadata-preservation patch. Direct rating refs and
  intake-built rating specs now preserve discovered policy metadata into the
  durable rating input, not only at live load time.

## 2026-05-15 E2E Validation Phase

Historical after the 2026-05-16 00:54 EDT purge unless a row explicitly names
`curvy-r18fresh-*`. Keep these notes for lessons and code provenance, not as
current remote state.

- Current phase: deployed end-to-end validation and hardening. Stop broad
  design churn unless it removes an immediate proof or launch blocker.
- Fresh large-bootstrap launch is now running. Manifest
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18v2-bootstrap-20260516a/curvy-r18v2-bootstrap-20260516a.json`
  was submitted to deployed app
  `curvyzero-lightzero-curvytron-visual-survival-train-v2` after syntax audit,
  Modal ref audit, grouped dry-run, focused tests, and trainer/tournament
  redeploy checks. Submission wrote `3` immutable control assignments,
  `3` control refresh pointers, and spawned `18` trainers plus `18` pollers.
  Immediate status showed trainer heartbeats and pollers running but no
  checkpoints yet. Next proof gate: see real progress/checkpoints from this
  batch, seed live v2 tournament intake from the exact run ids, rate those
  checkpoints, publish/materialize a valid assignment, and see the same running
  trainers apply/use it.
- First live-batch status at about `2026-05-16T01:41Z`: all `18` rows have
  `iteration_0`, all pollers are running, and all `18` rows have at least one
  live checkpoint eval/GIF artifact. Latest eval mean survival at startup is
  roughly `0..170` steps depending on row/opponent sample; this is only a
  baseline, not a learning claim.
- Live tournament watch is seeded:
  `curvy-r18v2-bootstrap-live-20260516a` /
  `elo-r18v2-bootstrap-live-20260516a`. Seed used exact run ids with
  `checkpoint_selection=all`, found `18/18` checkpoints, missed `0`, enqueued
  `18` events, and spawned rating call
  `fc-01KRQ78KSP09GH0R4K5B4YH3QE`. Settings are all-v2, all-pairs,
  `21` games per pair, one-frame timing, `max_steps=1048576`, eval mode,
  `browser_lines + simple_symbols`, GIFs on, `5` samples per pair.
- Intake status after seed: queue length is `0`, so the seed events were
  consumed. Rating progress exists for `round-000000`: `18` players,
  `153` pairs, `3213` games, phase `games_running`; first progress read had
  `0` completed games. Next action: wait, then recheck progress before any
  publish/materialize attempt.
- Later same-batch status: all `18` trainers reached at least
  `iteration_10000`; many reached `iteration_20000`. Intake tick on the live
  tournament found `47` checkpoint refs with `0` missing. Intake status showed
  the manifest at `47` refs and queue length `3`. Rating progress then showed
  all `153` startup pairs had been started, with estimated seen games
  `3213/3213`, but no final pair summaries yet. Do not force a continuation
  while `round-000000` is still writing; wait for the first rating latest, then
  drain/continue from latest.
- 2026-05-16 current large-batch update: the `curvy-r18v2-bootstrap-20260516a`
  trainers are still alive. Every row has reached at least `iteration_10000`;
  most rows are at `iteration_20000` or `iteration_30000`; each row has `2-4`
  checkpoints plus eval/GIF artifacts. The live tournament is also alive.
  Durable rating output exists through `round-000002`: `37` checkpoints,
  `666` pairs, `13,986` games, continued from latest, and latest snapshot
  `stable=false` with `max_abs_delta=137.8483556121352`. `round-000003` is now
  running from a larger `64` checkpoint pool: `1,596` pairs and `33,516`
  planned games, with about `12,432` games started/seen by the cheap progress
  estimate. Intake status shows `64` seen/queued refs and queue length `1`.
  Plain meaning: the big loop is not dead; it is still rating a moving pool.
  Do not publish/materialize this large-batch leaderboard as training truth
  until a latest snapshot is `stable=true` and passes the public leaderboard
  gates.
- 2026-05-16 live-watch repair: an exact-ref submission had temporarily made
  the large intake scan explicit-ref only. Plain meaning: the service would
  remember the submitted checkpoint files, but it would stop discovering future
  checkpoints from the 18 live run ids. This is now repaired two ways. The
  running intake was re-seeded with the 18 run ids and
  `checkpoint_selection=all`; direct Volume config now shows `18` run ids,
  `checkpoint_selection=all`, `83` seen refs, and
  `continue_from_latest=true`. The code now preserves an existing live
  run-id/prefix watch whenever exact refs are pinned into the same intake.
  A second drain guard now refuses to spawn a continuation while a newer rating
  round input exists without its completed ratings file. Focused regression
  passed:
  `tests/test_curvytron_checkpoint_tournament.py::test_exact_refs_added_to_live_watch_preserve_run_scan`,
  `tests/test_curvytron_checkpoint_tournament.py::test_manifest_pool_merge_preserves_existing_live_watch_scan_spec`,
  and `tests/test_curvytron_checkpoint_intake_repair.py` (`20 passed`).
  Ruff passed for the touched files, and `curvyzero-checkpoint-tournament-v2`
  was redeployed from this code. Next live check: wait for the current large
  rating round to finish, then
  confirm the next continuation includes the larger live pool before any
  publish/materialize step.
- 2026-05-16 latest large-batch state: `round-000003` has finished and is
  still unstable (`57` rated checkpoints, `stable=false`). The live watch now
  has `92` checkpoint refs from the 18 running trainers. A stale
  `round-000004/progress.json` placeholder exposed two real bugs: a zero-work
  "waiting for input" progress file could block the round writer, and a
  detached rating loop used `.remote()` for child rounds. Both are patched,
  focused tests pass, and `curvyzero-checkpoint-tournament-v2` was redeployed.
  `round-000004/input.json` now exists and real games are running with
  balanced random seats and `max_steps=1048576`. Progress at
  `2026-05-16T03:02Z`: `4,186` pairs, `87,906` games planned, `886` pairs
  started, about `18,606` games seen by shard-summary estimate, `0` estimated
  failures. Logs from the active detached worker
  `ap-t8dhK6PpMxvqhyGo6hMRrG` show successful games, balanced random seat
  order, checkpoint ids from current r18v2 rows, and the intended high max-step
  cap. `latest.json` still points at unstable `round-000003`, so no
  publish/materialize step is allowed yet. Focused/broad tournament-intake
  regression now passes (`162 passed, 11 skipped`) and ruff passes for the
  touched files. Do not publish/materialize from this large lane until the
  latest snapshot is `stable=true`.
- Current survival readout is mixed, not a clean learning win. The latest eval
  means span roughly `98.25` to `204.75` steps. Several rows improved at some
  intermediate checkpoint, but many latest checkpoints fell back, and some rows
  show action-collapse warnings. Treat this as liveness and policy-change
  evidence only; survival improvement remains unproven.
- 2026-05-16 canary proof update: the warningfix canary closed the important
  trainer refresh proof. Tiny tournament `curvy-e2e-warningfix-live-20260516a`
  / `elo-e2e-warningfix-live-20260516a` rated two checkpoints with `21` games,
  `0` failures, and `stable=true`, then published public leaderboard snapshot
  `e2e-warningfix-live-r0-20260516a`. Promotion wrote assignment sha
  `8b171c177c401b886a5658fafc1c16076b5797c640b6d6a689003575e6d46208` and
  rewrote the running canary's control pointer. The same trainer
  `curvy-e2e-warningfix-canary-20260516a` applied that sha at train iter
  `5693` with `env_ready_report.ok=true`. Its `env_steps.jsonl` then contained
  `87` rows with the new assignment sha, all `87` with
  `opponent_provider_load_ok=true`, and no observed
  `opponent_provider_load_ok=false` rows. This proves
  checkpoint -> tiny tournament -> public leaderboard -> assignment/pointer ->
  same-trainer refresh -> env use, at canary scale.
- Do not overclaim that proof. It does not prove survival improvement, a
  production-quality ranking, or the large 18-run loop under load. It does
  prove the refresh plumbing works after the warning-leak fix.
- 2026-05-16 local/deploy verification after the slot-contract cleanup: focused
  tests passed (`131 passed, 3 skipped` for opponent/plumbing/publisher slices;
  `26 passed` for learner-seat, no-op action, and tonight18 manifest slices),
  and ruff passed for the touched files. The v2 trainer app and v2 tournament
  app were redeployed from the current code after those checks:
  `curvyzero-lightzero-curvytron-visual-survival-train-v2` and
  `curvyzero-checkpoint-tournament-v2`.
- 2026-05-16 current-code live-watch canary is now running to close the
  subscriber/intake side on the exact deployed code. Manifest:
  `artifacts/local/curvytron_e2e_canary/curvy-e2e-currentlive-canary-20260516a/manifest.json`.
  Submission spawned train call `fc-01KRQ4B3MMWSX44VS8ENSDAQM5` and poller call
  `fc-01KRQ4B3FKT7VS3QZA2AFQQHY7` on
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`, with starter
  assignment sha
  `af4167dd73868e5d7444b4b40b7ef28c86f6cfdc71c41f62a5cda368e02df81f`.
  First status showed the trainer alive with checkpoints through at least
  `iteration_1250` and refresh checks firing as `unchanged` on the starter sha.
- Current-code live-watch tournament was seeded from the canary run id, not
  explicit refs: `curvy-e2e-currentlive-live-20260516a` /
  `elo-e2e-currentlive-live-20260516a`. The seed discovered `34` checkpoint
  refs from `curvy-e2e-currentlive-canary-20260516a`, enqueued `34`
  checkpoint-seen events, preserved sidecar policy metadata
  (`cpu_oracle`, `browser_lines + simple_symbols`, `random_per_episode`, one
  source frame), and spawned rating call `fc-01KRQ4E0P30S5N6HECHXYQ8VPJ`.
  Round 0 started with `630` all-pairs battles and `13,230` planned games.
  This is useful live subscriber/intake proof and stress evidence, but it is
  larger than needed for the final same-trainer refresh proof. If it runs too
  long, start a smaller honest proof in parallel instead of waiting on it.
- Smaller honest proof started in parallel:
  `curvy-e2e-currentlive-sparse-canary-20260516a` /
  `try-e2e-currentlive-sparse-canary-20260516a`. This run saves checkpoints
  every `1000` train iterations instead of every `50`, so a live run-id intake
  seeded after `iteration_1000` should have only two useful checkpoints and a
  `1 pair * 21 games` tournament. Dry-run and Modal ref audit passed; launch
  spawned train call `fc-01KRQ4K9RGG7P5VAQBS9H19Z3W` and poller call
  `fc-01KRQ4K9N4M9WMA0SGWF5D56GW`, with starter assignment sha
  `4b7261e7a795da17517360a768b581879dd20ffa034d30f8cc7540858a731f4b`.
  First status at about `6.6s` elapsed showed heartbeat/progress alive,
  background poller running, one unchanged assignment-refresh event, and no
  checkpoints yet. Next action: wait for `iteration_1000`, then seed
  `curvy-e2e-currentlive-sparse-live-20260516a` /
  `elo-e2e-currentlive-sparse-live-20260516a` from the run id.
- Sparse canary status then reached train iter `1756` with exactly two
  checkpoints, `iteration_0` and `iteration_1000`, and the trainer still
  running. Sparse live-watch intake was seeded from the run id:
  `curvy-e2e-currentlive-sparse-live-20260516a` /
  `elo-e2e-currentlive-sparse-live-20260516a`. It found exactly `2`
  checkpoints, wrote `2` queue events, preserved policy sidecar metadata, and
  spawned rating call `fc-01KRQ4VESF8BRCWYY763KDG09S`. Round 0 is the intended
  tiny shape: `1` pair and `21` planned games.
- Sparse live-watch rating finished at `round-000007`: `1` pair, `21` games,
  `0` failed games, `stable=true`, `max_abs_delta=4.060196778430208`, and
  `ratings_written=true`. Promotion succeeded: public leaderboard
  `e2e-currentlive-sparse-live-r7-20260516a` snapshot sha
  `a1a003523adde3f3fc273ecba9b825da60f3c75f0b7ff23064e2db34ea24a79b`;
  assignment ref
  `control:training/lightzero-curvytron-visual-survival/e2e-currentlive-sparse-promotion-bank-20260516a/attempts/try-e2e-currentlive-sparse-promotion-bank-20260516a/opponents/assignments/e2e-currentlive-sparse-live-r7-assignment-20260516a/assignment.json`;
  assignment sha
  `774b70dd15fa71bc59a92819f3d417c9025184d6a24634ad4dbebe490dbb1009`;
  pointer
  `control:training/lightzero-curvytron-visual-survival/e2e-currentlive-sparse-control-20260516a/attempts/try-e2e-currentlive-sparse-control-20260516a/opponents/current_assignment_pointer.json`
  was rewritten. The same sparse trainer then applied this sha at train iter
  `5373` with `env_ready_report.ok=true`. A later env telemetry fetch contained
  `357` env-step rows with the promoted sha, including `312` rows with
  `opponent_provider_load_ok=true` and `0` observed rows with
  `opponent_provider_load_ok=false`. This closes the current-code live run-id
  intake -> tournament -> leaderboard -> assignment -> same-trainer refresh ->
  provider-ok env-use proof at canary scale.
- Follow-up live status, 2026-05-16: sparse live-watch status later showed `6`
  seen/queued checkpoints through `iteration_5000`, queue length `0`, trainer
  still running on the promoted assignment, and background evals for six
  checkpoints. Tiny one-seed eval mean steps by checkpoint were
  `342, 290, 413, 601, 249, 561`; this is noisy canary evidence, not survival
  proof, but it shows non-collapsed policy behavior and continuing checkpoint
  discovery.
- The larger high-frequency current-live stress canary
  `curvy-e2e-currentlive-canary-20260516a` is not the launch proof. It produced
  `77` checkpoints and `65` eval manifests, live intake saw `67` checkpoints,
  but the trainer never applied a promoted assignment. Its status was marked
  failed because the post-run artifact scanner looked for relative
  `training/.../lightzero_exp` paths outside the mounted runs Volume even though
  LightZero returned `ok=true` and saved checkpoints. Local fix now makes the
  scanner also check `RUNS_MOUNT / exp_name`; regression
  `test_scan_lightzero_artifacts_checks_runs_mount_for_relative_exp_name` passed
  and the full live-checkpoint plumbing file passed (`88 passed, 3 skipped`).
  Treat this canary as stress/discovery evidence only.
- Current correction: do not wait for a perfect starting ranking before
  bootstrap training. The optional ranked rerate is only for high-quality
  leaderboard-derived top slots. Bootstrap can use curated exact checkpoint
  refs, immortal blank/hard-coded pressure, and live tournament intake.
- Plain language correction: there is no such launch blocker as a "stable
  source leaderboard" for bootstrap. A stable, hash-guarded leaderboard matters
  only when the Coach wants to materialize leaderboard-derived opponent slots.
  The current bootstrap batch is allowed to learn from exact refs and explicit
  sentinel pressure while the tournament catches up.
- Even plainer: a "source leaderboard" is just a ranked list used to choose
  better starting frozen checkpoint opponents. It is not part of the core loop
  and it must not block the core loop proof. The core loop can start with old
  exact checkpoint refs, immortal blank/hard-coded sentinels, and whatever new
  checkpoints the trainers produce. If a better old champion is found later, it
  can be injected later.
- Current opponent-pressure rule: blank and hard-coded sentinel slots are
  immortal all the time. Leaderboard/checkpoint slots are mortal most of the
  time, with only small explicit immortal slices when a recipe asks for them.
  Keep total immortal opponent exposure around `20-30%`, generally not above
  `30%`.
- Current implementation audit: two independent read-only audits confirmed this
  is what the code does. Generic mixtures reject `blank_canvas_noop`,
  `fixed_straight`, and `proactive_wall_avoidant` training entries unless they
  use `opponent_immortal=true`; frozen checkpoint entries can be duplicated as
  small explicit immortal slices; learner seats default to `random_per_episode`;
  tournament seats default to balanced random physical seating; action `1` is
  the explicit straight/no-turn action for live players.
- Do not use `stable_slots_v1` as the next bootstrap path if immortal checkpoint
  slices are required. That materializer intentionally keeps checkpoint slots
  mortal and rejects a bulk "make all checkpoints immortal" switch. The current
  bootstrap path is the manifest recipe path, which expresses immortality with
  explicit weighted mixture entries.
- Historical bootstrap manifest prepared locally:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18bootrefs-20260515a/curvy-r18bootrefs-20260515a.json`.
  It was built from
  `artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top96-nonzero-20260515a/refs.txt`
  with `--checkpoint-refs-file`, not from a trusted ranking. It has 18 rows,
  three control-volume assignment recipes, three control-volume refresh
  pointers, `random_per_episode`, shared initial checkpoint
  `iteration_240000.pth.tar`, and total immortal pressure of `20%`, `25%`, and
  `30%` across the three recipes. Dry-run submitter passed. Manifest ref audit
  against `curvyzero-runs-v2` passed with `4/4` exact refs present and `0`
  missing/bad refs.
- Fresh current-code bootstrap review artifact, not launched:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18v2-bootstrap-20260516a/curvy-r18v2-bootstrap-20260516a.json`.
  It has 18 rows, all-v2 app/Volume names, `random_per_episode`, control-volume
  assignment refs and refresh pointers, `assignment_refresh_interval_train_iter=2000`,
  `save_ckpt_after_iter=10000`, `browser_lines + simple_symbols`, `gpu-h100-cpu40`,
  and exact refs from
  `artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top96-nonzero-20260515a/refs.txt`.
  Local syntax audit passed with `4` exact refs and `0` bad refs; Modal existence
  audit against `curvyzero-runs-v2` passed with all `4/4` refs present; grouped
  submitter dry-run selected all 18 rows and would write exactly 3 assignments
  plus 3 refresh pointers before train/poller spawns.
- Current slot cleanup after side audit: generic opponent mixtures now reject
  hard-coded entries (`fixed_straight` and `proactive_wall_avoidant`) unless
  they set `opponent_immortal=true`. The old stable-slot selector no longer
  accepts a bulk "make every checkpoint slot immortal" flag; frozen checkpoint
  immortality should be represented as explicit small mixture slices. Focused
  tests: `uv run pytest tests/test_opponent_mixture.py tests/test_opponent_leaderboard.py -q`
  -> `42 passed`; ruff passed for touched files.
- Launch attempt `curvy-r18bootrefs-20260515a` exposed a real control-volume
  visibility race: assignment files were written and later visible in
  `curvyzero-curvytron-control-v2`, but warm trainer containers tried to read
  them before reloading their mounted view and crashed with
  `opponent assignment file not found`. Fix landed locally: trainer startup now
  reloads the referenced control/runs volume before the first assignment read
  and also reloads checkpoint volumes for assignment-backed frozen opponents.
  Focused regression passed; redeploy/relaunch is next.
- Relaunch after the fix is now live as `curvy-r18bootfix-20260515a`.
  Validation before launch: dry-run submitter passed, syntax audit passed, and
  Modal ref audit against `curvyzero-runs-v2` passed with `4/4` exact refs
  present. Submission wrote `3` control-volume assignments, `3` control-volume
  refresh pointers, and spawned `18` trainer calls plus `18` poller calls.
  First status read after launch showed all `18` rows with train heartbeats and
  background pollers `running`; no assignment-file crash repeated. No
  checkpoints yet at that first read, so the batch-level full loop is still in
  monitoring, not proven.
- Later status read for `curvy-r18bootfix-20260515a`: all `18` rows had at
  least `iteration_0` checkpoints and poller activity. That proves launch and
  first checkpoint writing, not the full loop. The next needed proof is a
  nonzero checkpoint from this batch admitted into a live v2 tournament, rated,
  materialized into a new assignment/pointer, and then used by a running trainer
  with provider-ok env telemetry.
- Follow-up status read showed the status table can be misleading: the
  displayed checkpoint was still `iteration_0`, but a direct
  `progress_latest.json` read for a sampled row showed
  `learner_train_iter=7037` at `elapsed_sec=668`. The jobs were training toward
  the first `save_ckpt_after_iter=10000` checkpoint rather than dead.
- Latest tournament discovery over `curvy-r18bootfix-` found all `18` current
  rows and `0` missing refs. At first read, `7/18` latest refs had reached
  `iteration_10000.pth.tar`; at the later pre-intake read, `13/18` latest refs
  had reached `iteration_10000.pth.tar` and the other `5/18` latest refs were
  `iteration_0.pth.tar`. The discovered refs carried the expected sidecar
  metadata: `policy_observation_backend=cpu_oracle`,
  `policy_trail_render_mode=browser_lines`,
  `policy_bonus_render_mode=simple_symbols`,
  `learner_seat_mode=random_per_episode`, and
  `decision_source_frames=1`. This is liveness and metadata proof, not yet
  full-loop proof.
- Live v2 tournament intake has now been seeded for the larger batch:
  `curvy-r18bootfix-live-20260515a` /
  `elo-r18bootfix-live-20260515a`. The seed found and enqueued `32`
  checkpoints from the `curvy-r18bootfix-` prefix with
  `checkpoint_selection=all`, all-v2 volumes, `pair_selection=all_pairs`,
  `games_per_pair=21`, `save_gif=true`,
  `gif_sample_games_per_pair=5`, `max_steps=1048576`,
  `policy_mode=eval`, `browser_lines + simple_symbols`, and one source frame.
  The rating round exists and completed: `round-000000` had `32` checkpoints,
  `496` all-pairs battles, `10,416` planned games, `0` failed games,
  `ratings_written=true`, `stable=false`, and
  `max_abs_delta=117.51743133112922`. Plain meaning: this proves the tournament
  can ingest and rate current-batch checkpoints, but the resulting ranking is
  not final truth from one pass. Logs show real game summaries with balanced
  random seats and `max_steps=1048576`.
- Real refresh bug found from the same status read: the first refresh check
  tried to reload the runs Volume while TensorBoard event files were open, so it
  logged `kept_previous` with a visible `volume reload failed` reason. Training
  continued, but the old deployed code could keep failing refresh checks. Local
  fix: assignment refresh keeps control-volume reload strong, treats runs-volume
  checkpoint reload as best-effort, tries the existing mount anyway, and only
  keeps the previous assignment if the new checkpoint truly is not readable.
  Focused tests and ruff passed; `curvyzero-lightzero-curvytron-visual-survival-train-v2`
  has been redeployed with the fix. Existing running jobs keep their old code;
  a fresh canary or relaunch is required to prove the remote refresh fix.
- False-proof warning: the active `curvy-r18bootfix-` trainers were launched
  before the latest trainer refresh reload patch. Their checkpoints are still
  useful for tournament/intake proof, but they do not prove the patched trainer
  can consume a refreshed tournament assignment. Close that with a fresh patched
  canary or a relaunch, and require post-refresh env telemetry with
  `opponent_provider_load_ok=true`.
- Fresh patched refresh canary is now running:
  `curvy-e2e-patched-refresh-canary-20260515a`, starter assignment sha
  `9ebcba51ffa02e7ef39efb06cd7feebaae457901465c47807d17712601ee0ea7`.
  The canary reached numbered checkpoints; live intake
  `curvy-e2e-patched-refresh-live-20260515a` /
  `elo-e2e-patched-refresh-live-20260515a` admitted those checkpoints with the
  expected sidecar metadata. Its active `round-000000` started with `26`
  checkpoints, `325` all-pairs battles, and `975` planned games; round 0
  completed with `0` failures but was `stable=false`, so publication correctly
  refused. A frozen two-checkpoint intake was then started as
  `curvy-e2e-patched-refresh-frozen2-20260515a` /
  `elo-e2e-patched-refresh-frozen2-20260515a`; final `round-000007` is stable
  with `1` pair, `21` games, and `0` failures. Promotion succeeded and rewrote
  the canary pointer to assignment sha
  `f8a469b5ff8598fe64bd42906de64fb68d06a8aa75f6f4a2c20be82fa4c8eedc`.
  Remaining proof: same running patched trainer must log `decision=applied`
  for that sha and then env rows with `opponent_provider_load_ok=true`.
- Latest ranked-source audit: the recreated all-v2 tournament volume
  currently contains only the tiny all-v2 canary arena
  `curvy-e2e-allv2-canary-live-20260515a` and one public leaderboard
  `e2e-allv2-canary-live-r3-20260515a`. That snapshot is valid wiring proof
  only: `25` rows, `4` active rows, relaxed/provisional gates, and rank 1 is
  `iteration_0.pth.tar`. Do not use it as a production-quality
  leaderboard-derived opponent source.
- Current optional ranked-source status is now plain: a trusted ranked source
  needs either a fresh production-shaped all-v2 rating/leaderboard snapshot, or
  an explicit rematerialization/rerate step that copies selected old checkpoint
  refs into v2 storage and rates them there. This is useful for high-quality
  leaderboard slots, but it must never block bootstrap/static training from
  curated assignments and exact refs.
- Plain correction after reorientation: do not use "stable source leaderboard"
  as a fake blocker. It only means "a ranked source good enough to choose
  leaderboard-derived frozen slots." Bootstrap and loop validation can run from
  exact old checkpoint refs, immortal blank/hard-coded sentinels, and new
  checkpoints as they arrive. Better old winners can be injected later.
- Current slot rule: blank/hard-coded sentinel slots are immortal all the time.
  Frozen checkpoint/leaderboard slots should usually be mortal, with explicit
  small immortal slices if the recipe needs more pressure. Keep total immortal
  exposure around `20-30%`, generally not above `30%`.
- Local refresh hardening just landed: non-fatal checkpoint Volume reload
  warnings are no longer inserted into `opponent_mixture.entries`. They are
  returned as resolved-assignment metadata instead, so the env-facing slot
  object stays valid. Regression:
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_resolve_opponent_assignment_checkpoint_reload_failure_can_be_nonfatal -q`
  -> `1 passed`; broader slice
  `uv run pytest tests/test_opponent_mixture.py tests/test_curvytron_live_checkpoint_eval_plumbing.py -q`
  -> `106 passed, 3 skipped`; ruff passed for the touched trainer/test files.
  The v2 trainer app was redeployed from this patch at about 2026-05-16
  00:xx EDT. New trainer calls use the fix; old already-running jobs do not.
  Fresh deployed canary `curvy-e2e-warningfix-canary-20260516a` has launched
  from the new image: train call `fc-01KRQ2NT07PHZMHZ05N2DRKC3M`, poller call
  `fc-01KRQ2NSWRK3MS5JYPKXJB5EJ3`, starter assignment sha
  `1706dc31744865da9f1d1ff8acebebd7aa815f61e815fedeed7b6b6c92ee3cc9`.
  First status showed the train heartbeat alive at stage `auto_resume_checked`,
  but no progress/checkpoints yet. Second status at about `00:28Z` showed the
  run alive with `15` checkpoints, latest `iteration_1400`, evals through
  `iteration_1000`, and `4` refresh events all `unchanged` on the starter sha.
  Next proof: promote a tiny assignment into the pointer, then require
  `decision=applied` and provider-ok env rows from this same run.
- Current recommended optional source strategy: use the old
  `loop18-main-adaptive417` leaderboard only to select candidate refs, copy the
  top active exact checkpoints into `curvyzero-runs-v2`, then run a fresh v2
  tournament/rating under new ids. Do not copy the old leaderboard as truth.
- Concrete source-ref plan generated:
  `artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top100-20260515a/`.
  It selected `100` active refs from the old `loop18-main-adaptive417`
  snapshot, wrote `refs.txt`, `selection.json`, and `commands.txt`, and plans a
  fresh rerate as `curvy-restart18-source-rerate-20260515a` /
  `elo-restart18-source-rerate-20260515a`. The selected pool has `4`
  `iteration_0` refs; acceptable as candidate-pool noise for rerating, but do
  not mistake it for a trusted leaderboard-derived opponent source.
- Source-ref audits now make the copy gate explicit:
  `source-refs-old-volume-audit.json` proves all `100/100` selected refs exist
  in historical `curvyzero-runs`; `source-refs-v2-target-before-copy-audit.json`
  proves `0/100` currently exist in `curvyzero-runs-v2`. That is expected before
  rematerialization. After copying, rerun the generated target audit command and
  require `100/100` present in v2 before starting the fresh source rerate.
- Rematerialization is now done: `rematerialize-report.json` says `100/100`
  refs copied from historical `curvyzero-runs` into `curvyzero-runs-v2`, and
  `source-refs-v2-target-after-copy-audit.json` proves `100/100` refs now exist
  in v2 with `0` bad refs and `0` missing refs.
- Fresh all-v2 source rerate launched:
  `curvy-restart18-source-rerate-20260515a` /
  `elo-restart18-source-rerate-20260515a`, Modal call
  `fc-01KRPJE1C28EJZQK6VRYQ75JT7`, app run
  `ap-ll8SUOPBXNXKGCQmXhulOa`. Plan estimate: `100` checkpoints, `300` pairs,
  `6,300` games, `21` games per pair, `5` GIF samples per pair, eval mode,
  `browser_lines + simple_symbols`, `num_simulations=8`, one-frame timing,
  `max_steps=1048576`. First progress query at 2026-05-15 15:40 EDT showed
  `games_running`, `0/6300` games complete.
- Source rerate round 0 completed cleanly but is not publishable yet:
  `round-000000`, `300/300` pairs, `6300/6300` games, `0` recorded failures,
  `ratings_written=true`, `stable=false`, `max_abs_delta=32.58886751199381`.
  This means the first bounded source round ran, but the ratings are still
  moving too much to call it a trusted leaderboard-derived opponent source.
- Source rerate continuation was started with `--continue-from-latest`, Modal
  call `fc-01KRPJZ73XN3EQ0HA2CBPC3VWH`. Direct v2 Volume artifact check proves
  the continuation advanced to `round-000001`: `rounds/round-000001/input.json`
  exists with `6300` planned games.
- Source rerate round 1 also completed cleanly but is still not publishable:
  `round-000001`, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=25.065565057086832`. `latest.json` advanced to
  round 1, so continuation/reduction is working.
- Source rerate round 2 was launched as another detached continuation:
  Modal call `fc-01KRPKQYQJGGDKBYPME1KP20BZ`. The local CLI estimate showed
  zero checkpoints because the safer continuation launch passed no explicit
  refs, but the remote artifact is correct: `rounds/round-000002/input.json`
  has the full `100`-checkpoint roster, `previous_round_id=round-000001`, and
  `300` pairs / `6300` planned games. Trust the v2 Volume input artifact, not
  the local detached-launch estimate.
- Source rerate round 2 completed and advanced `latest.json`:
  `round-000002`, `300` pairs / `6300` games, `stable=false`,
  `max_abs_delta=23.31069784361553`. The per-round progress file still looked
  stale, so use `latest.json` plus `rounds/round-000002/ratings.json` as truth.
- Source rerate round 3 is running as detached continuation
  `fc-01KRPM5AS1TSSJMGQ961JYACBT`. The persisted input is healthy:
  `rounds/round-000003/input.json` has the full `100`-checkpoint roster,
  `previous_round_id=round-000002`, and `300` pairs / `6300` planned games.
- Source rerate round 3 completed and advanced `latest.json`:
  `round-000003`, `300` pairs / `6300` games, `stable=false`,
  `max_abs_delta=24.82819365645907`. This worsened slightly, so do not infer
  convergence from game completion alone.
- Source rerate round 4 completed cleanly and advanced `latest.json`:
  `round-000004`, `300` pairs / `6300` games, `stable=false`,
  `max_abs_delta=22.54218539334727`. The top row is now nonzero
  (`iteration_20000.pth.tar`), but rank 2 is still an `iteration_0` checkpoint,
  so this is not a production source yet.
- Current stability decision from the parallel scheduler critique: run a few
  more same-context `300`-pair adaptive continuation rounds. Do not lower K,
  raise `stop_max_delta`, switch to all-pairs, or publish `stable=false`.
  If rounds 4-6 stay around `20-25`, treat that as a scheduler/metric
  calibration issue and use a separate confirmation diagnostic instead of
  pretending the current threshold passed.
- Source rerate round 5 completed and advanced `latest.json`:
  `round-000005`, `stable=false`, `max_abs_delta=19.048764303143294`.
  Coverage is mature (`games_min=567`, `distinct_opponents_min=25`), but the
  source is still not publishable and still has four active `iteration_0` rows
  at ranks `2`, `3`, `7`, and `100`.
- Source rerate round 6 completed and advanced `latest.json`:
  `round-000006`, `300` pairs / `6300` games, `stable=false`,
  `max_abs_delta=18.39723682286698`. Coverage is mature
  (`games_min=714`, `distinct_opponents_min=29`), but the 100-ref lane now has
  `iteration_0` checkpoints at ranks `1`, `2`, `7`, and `100`. Treat this lane
  as diagnostic; do not use it as the restart source even if later rounds cross
  the numeric stability threshold.
- Nonzero fallback pool is prepared and audited:
  `artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top96-nonzero-20260515a/`
  has `96` refs with `iteration >= 1`, and
  both canonical source/target audits prove `96/96` exist in historical
  `curvyzero-runs` and current `curvyzero-runs-v2`. The fallback rerate is now
  launched as `curvy-restart18-source-rerate-nonzero-20260515a` /
  `elo-restart18-source-rerate-nonzero-20260515a`, Modal call
  `fc-01KRPN8G10S364JWVWC22VW0TP`. Round 0 completed `300` pairs /
  `6300` games, `stable=false`, `max_abs_delta=34.07017967162989`, with
  `96` rows but `0` active rows because the first placement pass gave top rows
  only `147` games and `7` distinct opponents. Round 1 completed
  `300` pairs / `6300` games with `0` failures, advanced `latest.json`, and is
  still not publishable: `stable=false`, `max_abs_delta=21.880940181012807`,
  `96` rows, `0` active rows, provisional coverage `231-294` games and
  `11-14` distinct opponents per checkpoint. That is expected because the
  rating active gate is `420` valid games and `20` distinct opponents for this
  96-checkpoint pool (`21` games per pair times `20` opponents). Round 2 is now
  launched as
  same-context continuation `fc-01KRPPAYWYA5T01JPZGQXCA9YH`, app
  `ap-0HzVT85O8UHt0rgUdMVyRg`; persisted `rounds/round-000002/input.json`
  is healthy with `previous_round_id=round-000001`, full roster preserved,
  and `300` pairs / `6300` games. Independent audit
  `restart18_nonzero_rerate_gate_audit_2026-05-15.md` projects that even a
  perfect round 2 can make at most `15/96` rows active because each checkpoint
  appears only `6-7` more times. Expect at least one more same-context placement
  round after round 2 unless the actual artifacts prove otherwise. Continue
  same-context rounds until the active gates are met and the stable gate passes.
- Round 2 completed and matched that projection: `round-000002`, `300` pairs /
  `6300` games, `0` failures in the final summary, `stable=false`,
  `max_abs_delta=22.572625403714373`, `96` rows, `15` active rows and `81`
  provisional rows. Coverage is now `357-420` games and `17-20` distinct
  opponents per checkpoint, so the source is still not publishable.
- Round 3 is now launched as same-context continuation
  `fc-01KRPPQA9JVBJTFWS1DYF2DB3D`, app `ap-91EOlo30iDhxlwDqVgJhYw`.
  Persisted `rounds/round-000003/input.json` is healthy:
  `previous_round_id=round-000002`, same `context_hash=3e1af9183db39818`,
  same `roster_hash=d2563608441af000`, `96` checkpoint roster entries, and
  `300` pairs / `6300` games.
- Round 3 completed and advanced `latest.json`: `round-000003`, `300` pairs /
  `6300` games, `rated_pair_count=300`, `invalid_pair_count=0`,
  `stable=false`, `max_abs_delta=39.7420779825474`, same
  `context_hash=3e1af9183db39818`, same `roster_hash=d2563608441af000`.
  Coverage is now mature (`96` active rows, `0` provisional rows, games
  `441-1008`, distinct opponents `21-48`, `0` failures), but the rating
  movement is too large to publish.
- Round 4 completed and advanced `latest.json`: `round-000004`, `300` pairs /
  `6300` games, `rated_pair_count=300`, `invalid_pair_count=0`,
  `stable=false`, `max_abs_delta=17.371056613899057`, same
  `context_hash=3e1af9183db39818`, same `roster_hash=d2563608441af000`.
  Coverage is mature (`96` active rows, `0` provisional rows, games
  `483-1575`, distinct opponents `23-73`, `0` failures), but the stability
  gate still blocks publish.
- Round 5 completed and advanced `latest.json`: `round-000005`, `300` pairs /
  `6300` games, `rated_pair_count=300`, `invalid_pair_count=0`,
  `stable=false`, `max_abs_delta=15.636412948237727`, same
  `context_hash=3e1af9183db39818`, same `roster_hash=d2563608441af000`.
  Coverage is mature (`96` active rows, `0` provisional rows, games
  `630-2058`, distinct opponents `28-89`, `0` failures). This is better but
  still not publishable.
- Round 6 completed as same-context continuation
  `fc-01KRPR02D72FZ27G5KED70GFBC`, app `ap-EqV1pzucLCW8fZjMA3FEqM`.
  Persisted `rounds/round-000006/input.json` was healthy:
  `previous_round_id=round-000005`, same `context_hash=3e1af9183db39818`,
  same `roster_hash=d2563608441af000`, `96` checkpoint roster entries,
  `300` pairs / `6300` games, `max_steps=1048576`, `policy_mode=eval`,
  `seat_order_mode=balanced_random`, `policy_trail_render_mode=browser_lines`,
  and `policy_bonus_render_mode=simple_symbols`.
- Round 6 final result: `300` pairs / `6300` games,
  `rated_pair_count=300`, `invalid_pair_count=0`, `0` failures,
  `96` active rows, `0` provisional rows, `stable=false`,
  `max_abs_delta=25.199213332028748`, same
  `context_hash=3e1af9183db39818`, same `roster_hash=d2563608441af000`.
  Coverage stayed mature (`games_min=693`, `games_max=2226`,
  `distinct_opponents_min=30`, `distinct_opponents_max=92`), but the
  stability delta worsened after round 5 (`15.636412948237727`).
- Current stability decision: do not publish this rerate for
  leaderboard-derived opponent assignment, and do not spend another blind
  same-context continuation round yet. First diagnose the round-6 max mover and
  scheduler exposure. The biggest mover was
  `ckpt-079-train-lightzero_exp-ckpt-iteration_240000-a391d866`, rank 21 ->
  rank 7, rating 1520.902 -> 1546.101, with `11` round-6 pairs / `231` games,
  `146` wins, `82` losses, `3` draws, no failures, and mostly `random_bridge`
  scheduling (`8` of `11` pairs). This may be a real strength correction or an
  adaptive-scheduler high-leverage swing; separate that before more rounds.
- Leaderboard-derived opponent publish gate: do not publish this rerate into a
  non-diagnostic training-facing leaderboard until latest is `stable=true`
  (`max_abs_delta <= stop_max_delta`, currently `10.0`), active rows are
  coverage-mature, and publish uses expected round/context/roster/snapshot
  hashes. Historical publisher behavior is not an active launch path: the
  current deployed/local publisher rejects unstable non-diagnostic training
  publishes. Diagnostic-only unstable snapshots are allowed only as evidence
  and must not steer opponent selection.
- New guardrail: `scripts/audit_curvytron_launch_manifest_refs.py` collects
  initial-policy refs and frozen assignment/mixture refs, then checks syntax
  and optional local/Modal existence. It now also accepts `--refs-file`, so the
  old-source and v2-target rematerialization checks do not need a launch
  manifest. The canary dry-run manifest passed a real Modal existence check
  against `curvyzero-runs-v2` with `4/4` unique refs present; that proves the
  guardrail, not launch quality.
- Shared Volume helper is stricter now:
  `modal_volume_kwargs_for_name("curvyzero-runs")` and other non-v2 Curvy
  volumes raise instead of silently returning old-volume kwargs. The explicit
  source-rematerialization script is the only current path allowed to read old
  `curvyzero-runs`, and it marks that audit with `--allow-non-v2-runs-volume`.
- Fresh all-v2 canary is proven at wiring scale:
  `curvy-e2e-allv2-canary-20260515a` /
  `try-e2e-allv2-canary-20260515a`; poller
  `fc-01KRPDXH9FW4DNNMW41G9XC3EY`; trainer
  `fc-01KRPDXHDF4C5RS17DR1Z18EVX`. The seed checkpoint was copied into
  `curvyzero-runs-v2` before launch, and the submitter now opens the v2 control
  volume with the shared VolumeFS-version helper. Proof result: the trainer
  wrote checkpoints through at least `iteration_5300.pth.tar`; v2 intake and
  tournament completed `round-000003` with `18/18` games, `0` failures, and
  `stable=true`; promotion wrote assignment sha
  `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0`; the same
  running trainer applied that sha at train iter `5061` with all `8` envs ready;
  a later refresh check stayed on the promoted sha at train iter `5372`;
  `env_steps.jsonl` had `1836` provider-ok rows using the promoted sha on the
  latest post-refresh fetch.
- What this still does not prove: production-quality promotion, a learned
  nonzero checkpoint as champion, background eval/GIF poller behavior, or
  survival improvement. The canary used relaxed/provisional gates and is a
  wiring proof only.
- Post-proof focused local regression passed. Latest tight guardrail rerun after
  refs-file audit and strict v2 Volume helper:
  `tests/test_rematerialize_curvytron_checkpoint_refs.py`,
  `tests/test_prepare_curvytron_restart_source_refs.py`,
  `tests/test_curvytron_launch_manifest_ref_audit.py`, and related strict-v2
  contract checks -> latest slices `17 passed` then `13 passed`; after the
  unstable-publish guard, `tests/test_curvytron_checkpoint_tournament.py` ->
  `134 passed, 11 skipped`; ruff passed on the touched tooling/contract/test
  files.
- Broader E2E-adjacent local regression passed after the all-v2 proof,
  fail-closed launch patch, and source-ref tooling:
  checkpoint tournament, live eval plumbing, source-state env, opponent
  leaderboard, opponent mixture, opponent registry, and GIF browser slices ->
  `352 passed, 24 skipped`.
- Current deployed trainer/tournament namespace after the latest redeploy is
  all-v2: `curvyzero-runs-v2`, `curvyzero-curvytron-control-v2`, and
  `curvyzero-curvytron-tournaments-v2`.
- Modal VolumeFS version check, 2026-05-15 14:08 EDT: verified with
  `Volume.from_name(name, version=2).info()` for `curvyzero-runs-v2`,
  `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`. Do not infer filesystem version from the
  name suffix alone; the code uses the explicit shared-contract map.
- Important correction from the latest remote smoke: the old
  `curvy-e2e-clean-canary-long-20260515c` assignment points at checkpoints in
  `curvyzero-runs`, so it is historical proof only. It must not be used as an
  input to the new all-v2 lane unless rematerialized into v2 storage.
- Pre-reset current-namespace poller smoke passed after writing a tiny control
  assignment in the then-current control volume that pointed at an actual
  checkpoint in `curvyzero-runs-v2`
  (`optimizer-gpuobs-canary-20260515/.../iteration_0.pth.tar`). Result:
  `curvy-e2e-poller-v2-namespace-smoke-20260515a` completed with
  `status=completed`, `seen_count=0`, and no assignment/checkpoint resolution
  error. This is useful historical evidence; the recreated all-v2 lane was
  later proven by `curvy-e2e-allv2-canary-20260515a`.
- Current proof status: trainer writes a fresh checkpoint -> intake/subscriber
  admits it -> tournament rates it -> public leaderboard/assignment is updated
  -> the same running trainer refreshes to the new assignment -> later trainer
  telemetry proves the refreshed opponent loaded and was used. This passed for
  the recreated all-v2 lane on `curvy-e2e-allv2-canary-20260515a`.
- The old `controlrun2` proof is real, but it predates the latest seat/slot
  cleanup. Treat it as a historical template. The current active proof is the
  all-v2 canary `curvy-e2e-allv2-canary-20260515a` above.
- Fresh restart/canary runs should use fresh all-v2 tournament/rating ids. The
  v2 queue was recreated empty, but fresh ids still keep proof artifacts
  readable and avoid accidental reuse of stale durable tournament folders.
- Canary shape: frequent checkpoints, short enough to observe quickly, deployed
  trainer/tournament apps, `control:` assignment pointer,
  `commit_on_checkpoint=true`, and `random_per_episode` learner seat.
- Pass/fail must stay written here and in `FULL_LOOP_PROOF.md` with exact run
  ids, assignment shas, checkpoint refs, and telemetry counts.
- First canary
  `curvy-e2e-clean-canary-20260515a` proved checkpoint production, live run-id
  intake, tournament rating, public leaderboard publication, assignment
  materialization, and control-pointer rewrite. It missed same-process refresh
  because the pointer was rewritten after the last refresh event and just before
  `max_train_iter=4000`.
- Long canary:
  `curvy-e2e-clean-canary-long-20260515c` /
  `try-e2e-clean-canary-long-20260515c`, train call
  `fc-01KRP6J8GDR05P6MYRZKXVQ9QK`, starting from short-canary champion
  `iteration_1000.pth.tar`, with `max_train_iter=20000`.
- Long-canary tournament/promotion is done:
  `curvy-e2e-clean-canary-long-live-20260515c` /
  `elo-e2e-clean-canary-long-live-20260515c`, `2` pairs / `6` games /
  `0` failures, assignment sha
  `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30`.
- Bug found and fixed: promotion wrote a duplicate pointer at literal
  `control:training/...` instead of replacing `training/...`. The script is
  patched, regression-tested, and the live pointer was manually repaired for the
  long canary.
- Current deployed proof result: passed on the long canary after pointer repair.
  The same running trainer applied sha
  `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30` at train
  iter `2750`, all `8` envs were ready, and `env_steps.jsonl` contained `1107`
  rows using that sha with provider load OK.
- Current remaining validation/hardening:
  keep the pointer-prefix regression test; record and later fix the
  duplicate-rating race; watch `/runs` inode pressure before any larger launch;
  separately remote-smoke background eval/GIF control-ref handling if the next
  launch depends on those pollers.
- Fresh hardening result after the canary:
  broad E2E-adjacent local suite passed with `385 passed, 14 skipped`;
  background eval poller control-ref assignment-file reload bug is patched and
  unit-tested; old long-canary training app was stopped after proof. The
  current active trainer app is now the all-v2 deployment
  `ap-TyUzvtvYjsO0YFrNx275CE` with `0` tasks.
- Fresh remote poller smoke found the next hardening gap: the poller can now
  read the `control:` assignment file, but if that assignment points at a
  `runs:`/`/runs` frozen checkpoint ref, the separate poller process may still
  miss the checkpoint file until the runs volume is explicitly reloaded before
  resolving/loading frozen opponents. This is a background eval/GIF poller
  blocker, not a contradiction of the same-trainer refresh proof above.
- Local patch after that smoke: assignment resolution now has a separate,
  poller-only nested-checkpoint reload flag. The active trainer refresh path
  still avoids broad `/runs` reload while training. Focused validation:
  `tests/test_curvytron_live_checkpoint_eval_plumbing.py` -> `85 passed,
  3 skipped`; ruff passed for the touched trainer/test files.
- Broad E2E-adjacent validation after the poller hardening passed:
  `386 passed, 14 skipped, 16 warnings`. This is pre-reset history; the fresh
  all-v2 deployed canary has now passed. The next remote step is launch
  hardening and larger-run readiness, not another hybrid smoke.

## 2026-05-15 Seat / Slot Recheck

- Current local truth: the trainer default is `random_per_episode`, fixed-seat
  modes are diagnostics, and old `ego_player_index` config is rejected.
- Current local truth: public opponent mixtures/assignments use
  `opponent_immortal`; raw public `opponent_death_mode` is rejected. Runtime
  selection adds the env-facing `opponent_death_mode` only after a slot is
  chosen.
- Current local truth: `blank_canvas_noop` public slots must set
  `opponent_immortal=true`; the blank sentinel in both leaderboard selectors
  now advertises that truth instead of relying on hidden env behavior.
- Current local truth: tournament eval specs pin policy observation surfaces
  with `policy_trail_render_mode` and `policy_bonus_render_mode`, default to
  eval/greedy policy mode, and use balanced physical seats.
- Current local truth: trainer run metadata writes the same policy observation
  surface onto checkpoints/run summaries (`policy_trail_render_mode`,
  `policy_bonus_render_mode`, `policy_observation_contract_id`, and
  `observation_contract`). Tournament checkpoint normalization reads that
  metadata so a checkpoint carries the observation surface it was trained on.
- Focused validation: `79 passed` for opponent slots, random learner seat,
  no-op/straight action contract, tonight18 manifest, and tournament eval
  parity slices; ruff passed for touched files.
- Side-agent red reports were rechecked against current code: checkpoint intake
  repair is now green locally (`17 passed`), and trainer surface metadata is
  now green locally (`11 passed`). Do not treat older failing snapshots of
  those suites as current.
- Checkpoint observation-surface readback was also spot-checked:
  `test_checkpoint_spec_reads_policy_render_mode_from_observation_contract`
  passed.
- Earlier cleanup inventory that said v2 durable objects were absent is now
  stale: the exact v2 Volumes/Dicts/Queue were deleted and recreated at about
  2026-05-15 14:08 EDT. Treat old cleanup statements as history, not launch
  truth.
- Fresh Modal inventory recheck after all-v2 cleanup/redeploy: tournament
  service `curvyzero-checkpoint-tournament-v2` is deployed, trainer service
  `curvyzero-lightzero-curvytron-visual-survival-train-v2` is deployed with
  `0` tasks, GIF browser `curvyzero-curvytron-gif-browser-v2` is deployed with
  `0` tasks, and the old non-v2 Curvy trainer/tournament deployments are
  stopped.
- Current deployed storage is all-v2. Non-v2 storage remains present only as
  historical evidence and must not be used for the next launch unless an
  explicit migration copies the needed artifact into the all-v2 lane.
- Fresh deployed Modal loop after the all-v2 reset is proven at canary scale by
  `curvy-e2e-allv2-canary-20260515a`: trainer checkpoint -> subscriber/intake
  -> tournament -> public leaderboard -> assignment refresh -> trainer uses the
  promoted opponent. Older non-v2 canaries remain historical mechanics proof
  only.

## 2026-05-15 Tournament File Cleanup

- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py` was too large
  and mixed too many jobs in one file.
- First safe split is landed locally. The public Modal entrypoint and function
  names stay the same.
- New modules:
  - `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament_settings.py`
  - `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament_runtime.py`
  - `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament_browser_render.py`
- The Modal image already copies all of `src/` into `/repo/src`, so these helper
  modules will deploy with the existing app packaging.
- Validation: focused browser slice `27 passed`; broader
  tournament/intake/leaderboard slice `153 passed, 11 skipped`; ruff passed.
- Next cleanup should move discovery, intake-manifest helpers, and rating
  artifact/progress helpers. Keep actual Modal functions and queue/dict mutation
  in the entrypoint until the helper modules are boring.

## 2026-05-15 Shared Contract Cleanup

- Current CurvyTron defaults now live in one file:
  `src/curvyzero/contracts/curvytron.py`.
- The old compatibility shim
  `src/curvyzero/infra/modal/curvytron_volume_names.py` is deleted. Do not
  reintroduce it.
- Current storage/app defaults are explicit shared-contract objects:
  `curvyzero-runs-v2` for trainer checkpoints/runs,
  `curvyzero-curvytron-tournaments-v2` for tournament artifacts,
  `curvyzero-curvytron-control-v2` for assignment/control files,
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-checkpoint-events-v2`, and
  `curvyzero-curvytron-opponent-leaderboard-live-v2`.
- Tournament UI "current" defaults also live in the shared contract. Current
  code points at the fresh all-v2 canary proof:
  `curvy-e2e-allv2-canary-live-20260515a` /
  `elo-e2e-allv2-canary-live-20260515a`. Older loop18/v2real18 arenas are
  historical/forensic until a fresh real launch id is chosen.
- `modal_volume_kwargs_for_name(...)` uses an explicit map of verified VolumeFS
  versions. It currently requests version 2 for `curvyzero-runs-v2`,
  `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`. Old non-v2 volumes are not current launch
  targets.
- Current app/Volume/Dict/Queue getters reject non-`-v2` overrides. If a shell
  or manifest tries to route the current CurvyTron lane to old names, that is a
  launch-time error, not a quiet compatibility path.
- Current training/tournament defaults are explicit: one source frame per
  action, `source_max_steps=1048576`, `save_ckpt_after_iter=10000`,
  `commit_on_checkpoint=true`, `learner_seat_mode=random_per_episode`, and
  policy observations are `browser_lines + simple_symbols`.
- The restart18 manifest builder now fails closed for real launches: it
  requires exactly one explicit source, either a ranked snapshot or a curated
  checkpoint refs file. Bootstrap should use `--checkpoint-refs-file` when we
  only need exact old checkpoint refs. It defaults to
  `opponent-source=assignment`, writes immutable assignments to the v2 control
  volume, creates per-recipe control-volume refresh pointers, and uses a coarse
  default refresh interval of `2000` learner train iterations. Inline-mixture
  manifests are diagnostic-only and must explicitly disable refresh.
- A diagnostic dry-run manifest was built from the all-v2 canary leaderboard to
  verify the builder/submitter fields. It is not launchable as the real batch:
  the source has only `4` active rows and rank 1 is `iteration_0.pth.tar`.
- Seat/perspective wording now has a single local contract:
  `policy_observation_perspective_contract_2026-05-15.md`. Coach/training owns
  learner-seat selection; Optimizer owns fast backends for the same
  controlled-player observation view.
- Learner perspective is now a first-class training setting. The current
  default is `random_per_episode`; each reset chooses the learned policy's
  physical seat deterministically from the episode/reset seed, and telemetry
  records both learner and opponent player indices.
- The action space is still exactly three actions: `left`, `straight`,
  `right`. `straight` is the explicit no-turn/no-op-in-turn-space action; do
  not add a fourth action unless the game contract itself changes.
- Opponent slots express immortality with public `opponent_immortal`. Fresh
  public mixture entries may not hand-author `opponent_death_mode`; episode
  selection derives that lower-level env switch. `blank_canvas_noop` entries
  must state `opponent_immortal=true` because the runtime treats them as
  inert/immune.
- `16.666666666666668ms` is not an arbitrary per-test magic number. It is
  `SOURCE_PHYSICS_STEP_MS * 1`, because the current contract is one action per
  source frame.
- Render parity work is deferred to the optimizer lane. The current launcher
  contract should not use `body_circles_fast` as the training policy surface;
  that path remains historical/diagnostic evidence only.
- Fresh tournament/rating specs now use `policy_trail_render_mode` and
  `policy_bonus_render_mode` for policy observations. Old aliases like
  `observation_*`, `source_state_*`, and generic `trail_render_mode` are not
  accepted as fresh spec inputs; old checkpoint metadata translation remains an
  explicit repair boundary.
- Focused validation after this cleanup:
  `uv run pytest tests/test_curvytron_shared_contracts.py
  tests/test_curvytron_survivaldiag_manifest.py
  tests/test_curvytron_opponent_mixture_manifest.py
  tests/test_curvytron_tonight18_manifest.py
  tests/test_curvytron_checkpoint_tournament.py::test_rating_context_hash_changes_for_evaluator_not_roster
  tests/test_curvytron_checkpoint_tournament.py::test_checkpoint_game_uses_per_seat_policy_modes_and_full_rich_gif
  tests/test_multiplayer_source_state_trainer_surface.py::test_body_circles_fast_is_rejected_by_current_trainer_surface
  -q` -> `35 passed`.
- Ruff passed on the touched contract, trainer, tournament, manifest, and
  focused test files.
- Focused validation after the blank/immortal public-slot cleanup:
  `uv run pytest tests/test_opponent_leaderboard.py tests/test_opponent_mixture.py
  tests/test_opponent_registry.py
  tests/test_source_state_visual_survival_learner_seat_regression.py
  tests/test_env_contract.py tests/test_curvytron_tonight18_manifest.py
  tests/test_curvytron_checkpoint_tournament.py::test_build_game_specs_randomizes_balanced_seat_order_by_default
  tests/test_curvytron_checkpoint_tournament.py::test_rating_counts_wins_from_each_games_actual_seat_order
  tests/test_curvytron_checkpoint_tournament.py::test_tournament_render_contract_pins_policy_surface_and_full_gif
  tests/test_curvytron_checkpoint_tournament.py::test_tournament_rejects_legacy_policy_surface
  tests/test_curvytron_checkpoint_tournament.py::test_checkpoint_game_uses_per_seat_policy_modes_and_full_rich_gif
  -q` -> `79 passed`.
- Ruff passed on the touched opponent leaderboard/mixture/registry tests and
  source files.

## 2026-05-15 Stop / Restart Decision

- 2026-05-15 10:45 EDT cleanup pass:
  - `opponent_leaderboard.py` no longer emits public stable-slot entries with
    `opponent_death_mode`; it emits `opponent_immortal` and lets episode
    selection derive the runtime env switch.
  - Focused validation: `uv run pytest tests/test_opponent_leaderboard.py
    tests/test_opponent_mixture.py tests/test_curvytron_tonight18_manifest.py
    -q` -> `48 passed`; ruff passed for touched assignment/mixture/manifest
    files.
  - Broader validation: `uv run pytest tests/test_opponent_leaderboard.py
    tests/test_opponent_mixture.py tests/test_curvytron_tonight18_manifest.py
    tests/test_source_state_visual_survival_learner_seat_regression.py
    tests/test_curvytron_live_checkpoint_eval_plumbing.py
    tests/test_curvytron_checkpoint_tournament.py
    tests/test_curvytron_tournament_scheduler_fairness.py -q` ->
    `271 passed, 14 skipped, 4 warnings`; ruff passed across the touched
    trainer/tournament/assignment files.
  - Older opponent-mixture manifest builder was also cleaned so `rf` means
    `body_circles_fast`, public mixtures use `opponent_immortal`, command/dry
    run kwargs satisfy the current grouped submitter, and public manifests no
    longer persist derived `opponent_death_mode`.
  - Final combined validation with the older mixture builder included:
    `282 passed, 14 skipped, 4 warnings`; ruff passed.
  - `cleanup_lane_2026-05-15.md` now treats v2 storage/control objects as exact
    purge candidates, not current evidence.
  - Exact v2 storage/control purge completed and verified absent: the v2 runs,
    tournament, control volumes, intake/leaderboard dicts, and checkpoint queue
    are gone. Non-v2 Curvy storage remains.
- Current decision: the live v2 real18 run is invalid enough to stop, not just
  label as weak evidence. It is useful only as smoke/history for diagnosing the
  failure modes.
- Current code direction: new training defaults to
  `learner_seat_mode=random_per_episode`; `fixed_player_0` and
  `fixed_player_1` are explicit diagnostics only. Old `ego_player_index`
  config is rejected.
- Current opponent-slot direction: slot recipes express death immunity with
  `opponent_immortal`. Policy kind, runtime mode, and immortality are separate
  ideas. `opponent_death_mode` may still appear as the env/runtime switch
  derived at episode selection, but it is not accepted as clean public slot
  intent.
- Tournament eval balanced physical seating is implemented locally by the
  tournament lane and covered by focused tests. Treat it as required for the
  next tournament deploy/restart.
- This former launch block is closed for the current code path: randomized
  learner seat/perspective handling, no-op/straight action semantics,
  tournament eval parity with the trainer surface, and stale-app/workspace
  cleanup have local proof or explicit launch-scoped decisions.
- Next manifest should use `random_per_episode` learner seat/perspective
  handling. Do not repeat the previous seat-0-only shape.
- Next manifest should globally include about `20-30%` immortal pressure.
  Blank/hard-coded sentinel slots are always immortal; checkpoint slots are
  mostly mortal with small explicit immortal slices. Do not reuse the previous
  weak `5%` wall-avoidant recipes as the main pressure plan, and do not exceed
  roughly `30%` total immortal exposure unless a future diagnostic is clearly
  labeled.
- Modal operating rule for the restart: prove durable behavior from deployed
  apps, use Volume JSON as truth, use Dict/Queue only for coordination, stop
  stale detached apps before trusting dashboards, and avoid designs that depend
  on broad Volume reloads during active file reads.

## 2026-05-15 09:40 EDT Reorientation

- New P0 risk: player perspective may be wrong or incomplete. We need to know
  whether the trainer ever trains the learned policy as seat 1 / player 2, and
  whether tournament games evaluate both seats in a way that matches training.
  Until this is audited, treat the corrected rerate as smoke evidence, not final
  proof that rankings are valid.
- Corrected 16.6667ms rerate
  `elo-v2real18-rerate67-allpairs-16ms-20260515a` is still useful as a live
  worker/timing/render smoke. Do not publish new training assignments from it
  until the perspective audit clears or a fix lands.
- The user is open to purging and relaunching, but the next launch should not
  repeat an invalid setup. First inventory the live trainers, checkpoints,
  survival/outcome metrics, leaderboard state, and weak-run intervention state.
- Weak-run immortal/blank intervention has not yet been proven applied. This is
  now a tracked lane, not a chat-only idea.
- Parallel agents now own five bounded audits: training player perspective,
  tournament evaluation perspective, live metrics, workspace cleanup, and
  weak-run immortal intervention.
- Main-thread rule for the rest of this incident: if a blocker appears, start a
  smaller honest parallel proof while debugging, but also stop or relabel any
  work that cannot answer the real question.
- Live metrics audit at this point: `21` tracked rows, `14` running, `7`
  failed, `215` checkpoints, max checkpoint `iteration_160000`. Old live
  leaderboard still has only `53` rows and max admitted iteration `30000`;
  corrected 16ms rerate is still running with successful game logs but no final
  corrected leaderboard yet.

## 2026-05-15 09:26 EDT Current Truth

- The v2 real18 trainers are alive enough to matter: latest tracked status had
  `21` rows (`18` original plus `3` replacements), `17` running, `4` failed
  originals, `90` durable checkpoints total, and latest checkpoint up to
  `iteration_60000`.
- The small deployed v2 loop is still the strongest full-loop proof: trainer
  checkpoint -> intake/tournament -> promoted assignment -> same running
  trainer refresh was observed with concrete assignment sha
  `adb04ed3905fb9c8984e5e213a9261079f0e4be188315912d12ae5290d55b770`.
- The current 67-checkpoint rerate
  `elo-v2real18-rerate67-allpairs-20260515a` is useful only as a liveness
  smoke. Do not promote it as final evidence. Review found that the tournament
  game runtime used `20.0ms` source ticks while the trainer/checkpoints use
  `16.666666666666668ms` source ticks.
- Immediate fix: patch the tournament contract so runtime timing mismatch is
  rejected, carry `policy_bonus_render_mode` explicitly beside
  `policy_trail_render_mode`, redeploy `curvyzero-checkpoint-tournament-v2`,
  and launch a fresh 16.6667ms rerate for the discovered v2real18 checkpoints.
- Local fix is now implemented and tested:
  `uv run pytest tests/test_curvytron_checkpoint_tournament.py
  tests/test_curvytron_gif_browser.py -q` -> `154 passed, 21 skipped`.
  Deploy and corrected rerate launch are next.
- Corrected tournament app is deployed. Corrected detached rerate is running as
  app `ap-MKU8vQNXqZWCqX6Dle0ztG`, function call
  `fc-01KRNXDVC9552230KK0KCBYZQ1`, rating id
  `elo-v2real18-rerate67-allpairs-16ms-20260515a`.
- Corrected rerate persisted input was verified: `67` checkpoints,
  `2,211` pairs / `46,431` games, `decision_source_frames=1`,
  `decision_ms=16.666666666666668`,
  `source_physics_step_ms=16.666666666666668`,
  `policy_trail_render_mode=body_circles_fast`,
  `policy_bonus_render_mode=simple_symbols`, `max_steps=1048576`,
  GIFs on with `5` samples per pair.
- Wrong-tick rerate app `ap-uIXpEjsU0Iy0lM0NHs8qEk` was stopped so it cannot
  waste compute or be mistaken for final evidence.
- Survival improvement is still not proven for the real18 lane. The best
  current proxy is weak: GIF physical-step mean moved about `121.6 -> 133.3`
  across comparable original rows, with latest greater than first in `8/16`
  rows. This is not a clean eval metric.

## Historical 2026-05-15 V2 Real18 State

- Invalidated historical v2real18 diagnostic batch:
  `curvy-v2real18-20260515a`, launched on
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`.
- Historical v2real18 tournament:
  `curvy-v2real18-live-20260515a` /
  `elo-v2real18-live-20260515a` on
  `curvyzero-checkpoint-tournament-v2`.
- Corrected tournament round `round-000001` completed cleanly:
  `231/231` pairs, `4,851/4,851` games, `0` failed games,
  `22` active rows, `0` provisional rows. `stable=false` only means ratings
  are still moving; the active-row gate is now passing.
- The earlier provisional-row issue was self-inflicted: `placement_min_games`
  was set to `420`, which is impossible for a 17-player all-pairs pool because
  each checkpoint can only get `16 * 21 = 336` games. The corrected run let the
  tournament choose the feasible active gate.
- Historical per-recipe assignment refresh was published from that invalidated
  v2real18 leaderboard:
  - `blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35` ->
    sha `9717c8b00d1e4a030026ca4188611f04d961b6d6a6f477f8758f11489d8f8d45`
  - `blank20-wall5-rank1_75` ->
    sha `e348714b7c960ea62423fd5a8cedaf20427778f764957a4142d5968bc2080f36`
  - `blank5-wall5-rank2_25-rank1_65` ->
    sha `4db8fe399ce6d423f50cb30d8269c2d18bbf1b7025f8c40ffb8163972604fb5a`
- Real-batch refresh proof is partial and live:
  status snapshot `run_status_after_refresh.json` showed `4/18` rows had
  already logged `decision=applied` with one of those new assignment hashes.
  `15/18` original rows were still running; `3/18` had failed.
- Failure diagnosis is concrete:
  the failed rows were using old deployed code where
  `lightzero_target_config.td_steps = source_max_steps = 1048576`. That value
  is a MuZero replay target horizon, not the game cap, and it triggered
  LightZero replay sampling failures: `ValueError: 'a' and 'p' must have same
  size`.
- Fix is implemented and deployed:
  the trainer no longer writes `td_steps` from `source_max_steps` for
  `source_state_fixed_opponent`, so LightZero keeps its stock `td_steps=5`
  while the environment can still allow long games.
- Replacements launched for the three failed rows from refreshed manifest
  `curvy-v2real18-refresh-r1-20260515a`:
  rows `r008`, `r009`, and `r011`. They use new run ids under
  `curvy-v2real18-refresh-r1-20260515a-*`, start from that historical
  v2real18 tournament's rank-1 checkpoint, and use the same control refresh
  pointers.
- Still unproven for the real batch:
  all live/replacement rows applying refreshed assignments, replacement rows
  surviving past the old replay crash point, background eval summaries finishing,
  and a quantified survival-improvement trend.

## 2026-05-15 06:57 EDT Live Check

- Current tournament latest snapshot is now `round-000003`: `40` rows, all
  `active`, `0` provisional. `stable=false` means ratings are still moving, not
  that the active-row gate is failing.
- Current tracked trainer set is `21` unique rows: `18` original real18 rows
  plus `3` replacements. Latest status read shows `17` running and `4` failed
  original rows.
- Checkpoint files discovered from those `21` run ids with
  `checkpoint_selection=all`: `67` refs across `20` runs; one replacement had
  no checkpoint at discovery time.
- The trainer side is ahead of the tournament side: status sees `61`
  checkpoint files, discovery sees `67`, but the tournament latest ranks only
  `40` rows. Immediate action is to submit the exact discovered refs to the
  current v2 intake in small batches, then run a detached drain/continuation and
  watch whether the tournament row count catches up.
- Assignment refresh uptake is real: `15/21` tracked rows have applied one of
  the three current tournament-derived assignment hashes at least once. The six
  without applied refresh are the three old failed originals plus the three
  replacement rows.
- Replay-crash replacement evidence is improving but not complete: replacement
  `r008` has reached `iteration_10000`; the other two replacements were still
  at startup checkpoints on this read.
- Survival improvement is still not proven for this real18 lane. Background
  eval manifest count is still `0`; GIF artifacts exist, but they are not a
  survival metric.

## 2026-05-15 07:12 EDT V2 Rerate Decision

- The first 67-ref continuation under the old rating id failed correctly:
  `latest snapshot context_hash does not match rating spec`. Cause: the manual
  seed changed the rating context from the old `policy_trail_render_mode=
  body_circles_fast` to null. We should not mix those histories under one
  rating id.
- The anti-shrink patch worked: after redeploy and reseed, both Dict and Volume
  intake manifests stayed at `67` refs with `placement_min_games=null` in the
  seed output. A later status check also showed the old latest moved from `40`
  to `53` active rows, but the old rating id is now historical for this
  purpose.
- Historical diagnostic rerate lane: for the exact 67 discovered refs, the old
  bounded v2real18 smoke explicitly set
  `policy_trail_render_mode=body_circles_fast`, high `max_steps=1048576`, GIFs
  on, and `all_pairs`. Keep this as timing/historical CPU-control evidence only;
  do not copy it into the restart target.

## 2026-05-15 07:20 EDT Live Check

- Fresh clean rerate is running under rating id
  `elo-v2real18-rerate67-allpairs-20260515a`, app id
  `ap-uIXpEjsU0Iy0lM0NHs8qEk`.
- The Volume `progress.json` for that rerate is stale at the initial
  `0/46,431` games read, but Modal logs are live and show successful game
  workers at pair indices in the hundreds. Sampled game logs show
  `ok=true`, `error_type=null`, and `max_steps=1048576`.
- Trainer status snapshot over `21` tracked rows:
  `17` running, `4` failed originals, `90` total durable checkpoint files,
  checkpoint range `0..7` per row, max latest checkpoint `iteration_60000`.
- Assignment refresh uptake is still real: `15/21` rows have applied one of
  the three current tournament-derived assignment hashes. The latest decision
  is usually `unchanged` because those rows already applied the same sha.
- Current applied assignment shas are evenly split across the three recipes:
  `4db8fe39...`, `9717c8b0...`, and `e348714b...`, five rows each.
- Eval/survival proof is still missing for this v2 real18 lane:
  `eval_manifest_count` is still `0`, and background eval completions are only
  `7` total. Do not claim survival improvement from checkpoint count or GIFs.

## 2026-05-15 05:09 EDT Reorientation

- The small `/control`-volume loop is the strongest proof we have: a trainer
  wrote checkpoints, intake accepted them, tournament rated them, promotion
  wrote a new assignment on `/control`, and the already-running trainer applied
  that new assignment.
- The longer behavior proof is not still cooking. Modal now shows
  `ap-ciAzi7ByfRueLxZLtqxuEf` stopped at `2026-05-15 05:05:35 EDT`. That run
  wrote checkpoints through at least `iteration_2000`, but it did not receive a
  second promoted assignment before the rating lane stalled.
- The longer proof's trainer side looked healthy. The earlier read that the
  tournament worker was stalled was stale. Fresh progress checks now show the
  long-proof ratings completed games.
- Cleanup agent Noether cataloged live Modal apps and stopped nothing. Reason:
  several app ids contain mixed stale and current tournament activity, so
  app-level stopping would be too blunt without a finer target.
- Current next direct test: run the smallest direct rating smoke with two exact
  checkpoints and `--wait`. If that completes, the game workers are okay and
  the bug is in intake/claim/background scheduling. If it stalls too, debug the
  rating/game worker path itself before launching more training.
- Direct rating separator result:
  `curvy-looplive-directrating-smoke-20260515a` /
  `elo-looplive-directrating-smoke-20260515a` completed. It rated
  `iteration_0` vs `iteration_3000` from
  `curvy-looplive-proof-controllong-20260515d`, played `3` games, wrote
  `latest.json`, and ended `stable=true`. This proves fresh direct rating and
  game workers can run. The remaining tournament blocker is narrower:
  intake/claim/background scheduling or stale round artifacts.
- Next separator:
  run the same two-checkpoint shape through `intake-seed` / `intake-drain`
  with `spawn_rating` and `wait`, using a fresh tournament/rating id.
- Intake separator result:
  `curvy-looplive-intake-smoke-20260515a` /
  `elo-looplive-intake-smoke-20260515a` completed through the intake path:
  `1/1` pair, `3/3` games, `0` failures, `stable=true`.
- Long proof rating correction:
  `elo-looplive-controllong-proof-fresh-20260515e` is complete:
  `10/10` pairs, `210/210` games, `0` failures. The stale
  `game_map_started` snapshot was written before the blocking map returned;
  it was not final state.
- Real remaining gap:
  the long trainer finished before we used the completed long-proof rating to
  publish a new assignment and refresh that same running trainer. We need a new
  behavior proof or real launch that keeps the trainer alive long enough for
  the tournament result to come back in.
- New behavior proof launch:
  `curvy-looplive-proof-controlrun2-20260515f` /
  `try-looplive-proof-controlrun2-20260515f` is now spawned on deployed trainer
  app `curvyzero-lightzero-curvytron-visual-survival-train-v2`, function call
  `fc-01KRNF20A3YCKANZEH8PV0G33F`. It uses the control assignment from the
  successful controlfast proof, starts from controlfast `iteration_135`, writes
  checkpoints every `50` train iterations with `commit_on_checkpoint=true`, and
  refreshes from a `/control` pointer every `25` train iterations. Goal: rate a
  fresh checkpoint and promote it back into this same still-running trainer.
- Controlrun2 behavior proof passed, 2026-05-15 05:30 EDT:
  checkpoint `iteration_400` from `curvy-looplive-proof-controlrun2-20260515f`
  was seeded through intake with the controlfast champion anchor. Rating
  `elo-looplive-controlrun2-proof-r0-20260515f` completed `1` pair / `3` games /
  `0` failures, `stable=true`. Promotion wrote `/control` assignment
  `looplive-controlrun2-proof-r0-assignment-20260515f`, sha
  `3ff1af447117e4e90cd1e82277530063d20ba14086d180df5474e7d5309dfa9d`, and
  rewrote the same running trainer's `/control` pointer. The trainer then
  logged `decision=applied` at train iter `1798`, `env_ready_report.ok=true`,
  and later env telemetry rows used the new sha with
  `opponent_provider_load_ok=true`. This proves the small deployed live loop.
- First post-refresh behavior read, 2026-05-15 05:34 EDT:
  `controlrun2` had `315` env telemetry rows and `67` terminal samples. Before
  refresh (`4fbc8...`) terminal samples were `49`, mean return `162.92`,
  median return `206`, win/loss `29/20`. After refresh (`3ff1...`) terminal
  samples were `18`, mean return `212.44`, median return `270`, win/loss
  `13/5`. This is a good canary sign, but the sample is too small to call
  survival improvement proven.
- V2 storage proof passed, 2026-05-15 05:56 EDT:
  `curvy-v2-looplive-proof3-20260515a` wrote v2 checkpoints, direct v2 rating
  `elo-v2-looplive-proof3-direct-r0-20260515a` completed `1` pair / `3` games /
  `0` failures, promotion wrote assignment sha
  `adb04ed3905fb9c8984e5e213a9261079f0e4be188315912d12ae5290d55b770` on the
  v2 control Volume, and the already-running v2 trainer applied it at train
  iter `1904`. Later env rows used that sha with
  `opponent_provider_load_ok=true`, including a slot pointing at the fresh
  proof checkpoint `iteration_300`.
- V2 intake recheck passed too: the intake-spawned rating
  `elo-v2-looplive-proof3-r0-20260515a` later showed `status=complete`,
  `1` pair / `3` games / `0` failures, `ratings_written=true`, `stable=true`.
  The earlier stuck read was another stale progress snapshot.
- First v2 post-refresh behavior read, 2026-05-15 05:58 EDT:
  `curvy-v2-looplive-proof3-20260515a` had `238` env rows and `50` terminal
  samples on the old assignment sha `d881...`, versus `117` env rows and `25`
  terminal samples on the refreshed sha `adb04...`. Mean terminal return moved
  from `118.24` to `134.72`; mean terminal length moved from `159.20` down to
  `144.96`. That proves the new assignment is being used, but it is not a clean
  survival-improvement claim.

## Current Live Objects

| Item | Value |
| --- | --- |
| Training app | `curvyzero-lightzero-curvytron-visual-survival-train-v2` |
| Tournament app | `curvyzero-checkpoint-tournament-v2` |
| GIF browser app | `curvyzero-curvytron-gif-browser-v2` |
| Current all-v2 proof tournament | `curvy-e2e-allv2-canary-live-20260515a` |
| Current all-v2 proof rating | `elo-e2e-allv2-canary-live-20260515a` |
| Current all-v2 proof training run | `curvy-e2e-allv2-canary-20260515a` |
| Diagnostic 100-ref rerate | `curvy-restart18-source-rerate-20260515a` / `elo-restart18-source-rerate-20260515a`; rejected as leaderboard-derived source because iteration-zero rows rose to the top |
| Current leaderboard-derived source candidate | `curvy-restart18-source-rerate-nonzero-20260515a` / `elo-restart18-source-rerate-nonzero-20260515a` |
| Historical clean tournament target | `curvy-loop18-live-main-20260514f` |
| Historical clean rating target | `elo-loop18-live-main-adaptive417-20260515b` |
| Historical training prefix | `curvy-n18conn-` |
| Stale manifest footgun | `artifacts/local/curvytron_tonight18_manifests/curvy-v2refresh18p-20260514b/curvy-v2refresh18p-20260514b.json` is not the current run source for checkpoint injection |
| Manifest builder | `scripts/build_curvytron_tonight18_manifest.py` |

## Current Truth

- Live reorientation, 2026-05-15:
  - Clean rating job `elo-loop18-live-main-adaptive417-20260515b` completed
    `300` pairs / `6,300` games and wrote `latest.json`.
  - `stable=false` means historical bounded evidence only. It is not a
    leaderboard-derived restart18 opponent source; that stricter path requires
    the fresh all-v2 source rerate to be `stable=true`, coverage-mature, and
    hash-guarded. Bootstrap/static training does not require this source and
    should not wait for it.
  - Promotion published the public leaderboard and wrote assignment
    `loop18-main-adaptive417-r0-assignment-20260515b`.
  - Consume smoke now passes with the immutable tournament winner as the
    initial policy checkpoint.
  - Latest verified smoke:
    `loop18-main-adaptive417-consume-smoke-20260515e`.
  - The trainer loaded
    `iteration_240000.pth.tar` from the rank-1 row into the MuZero model with
    `matching_shape`, skipped four mismatched reward/value head tensors, and
    preserved a fresh optimizer.
  - Smoke artifact verification reports:
    `initial_checkpoint_loaded=true`, `meaningful_model_load=true`,
    `fresh_optimizer_preserved=true`, `provider_ok_row_count=359`,
    `env_step_row_count=359`, `smoke_passed=true`.
  - This proves: clean tournament result -> public leaderboard snapshot ->
    immutable assignment -> trainer launch consumes assignment and starts from
    the tournament winner.
  - Updated proof, 2026-05-15:
    `curvy-looplive-proof-controlrun2-20260515f` proves a new checkpoint from a
    running trainer can be seeded through intake, rated, published/materialized,
    promoted to a `/control` assignment, refreshed into that same running
    trainer, and used by env workers.
  - The later all-v2 canary proves the v2 storage namespace is clean at wiring
    scale. Meaningful survival improvement over a long window and
    production-scale tournament quality remain separate evidence gaps.
  - Important correction: the existing 18 `curvy-n18conn-*` training runs were
    launched with fixed `opponent_assignment_ref` files only. Their manifest
    has no `initial_policy_checkpoint_ref`, no
    `opponent_assignment_refresh_interval_train_iter`, and no
    `opponent_assignment_refresh_ref`. They cannot prove the automatic live
    refresh loop.
  - Historical storage namespace footgun: the clean adaptive417 proof artifacts live in
    the non-v2 storage names:
    `curvyzero-runs`, `curvyzero-curvytron-tournaments`,
    `curvyzero-curvytron-control`, intake dict
    `curvyzero-curvytron-checkpoint-intake-v0`, and leaderboard dict
    `curvyzero-curvytron-opponent-leaderboard-live`.
    This was superseded by the all-v2 reset and all-v2 canary proof. Do not use
    non-v2 artifacts as current launch inputs unless explicitly copied into v2
    storage.
  - Trainer refresh patch, 2026-05-15:
    `opponent_assignment_refresh_ref` may now be either an immutable assignment
    JSON or a `curvyzero_opponent_assignment_refresh_pointer/v0` JSON containing
    `assignment_ref` and `assignment_sha256`.
  - Refresh pointer reload patch, 2026-05-15:
    the running trainer now calls `runs_volume.reload()` before reading the
    mutable refresh pointer. Without this, overwriting the pointer on the
    Volume might not become visible inside an already-running trainer.
  - Focused trainer tests after the pointer patch:
    `tests/test_curvytron_live_checkpoint_eval_plumbing.py` -> `79 passed,
    3 skipped`.
  - Historical trainer app redeployed after pointer patch with explicit non-v2
    proof storage env vars. The current active deployment is the all-v2 app
    namespace.
  - Live-loop proof failure found, 2026-05-15:
    `curvy-looplive-proof-fast-20260515a` wrote numbered checkpoints,
    intake discovered them, tournament rating completed, leaderboard was
    published, assignment was materialized, and the refresh pointer was
    overwritten. The running trainer did not apply the new assignment because
    `runs_volume.reload()` failed while the process cwd was inside `/runs`.
    A cwd-only patch was not enough: Modal also refuses to reload a Volume when
    LightZero has a TensorBoard event file open under that same Volume. Correct
    design is now separated: training logs/checkpoints stay on `/runs`, while
    mutable assignment pointers and refreshed assignments live on `/control`
    and use `control:` refs.
  - Control-volume patch, 2026-05-15:
    trainer functions mount both `/runs` and `/control`; `control:` refs reload
    only the control Volume; assignment writer can write to `target_volume=control`.
    Focused tests now pass with `83 passed, 3 skipped`.
  - Control-volume live proof, 2026-05-15:
    `curvy-looplive-proof-controlfast-20260515c` has now proven the full small
    loop. The trainer wrote checkpoints; intake accepted `129` refs; proof
    rating `elo-looplive-controlfast-proof-20260515c` completed `10` pairs /
    `210` games / `0` failures; promotion wrote a `/control` assignment with
    sha `4fbc8ef9d621ed5848a474d63f0cec900d6a97dd7827b5c1e81e17de0e12d462`;
    the already-running trainer refreshed at train iter `1130` with
    `decision=applied`, `env_ready_report.ok=true`, and later env rows used the
    same sha with provider load OK.
  - Remaining caveat:
    this is a small proof, not evidence that survival is improving in the large
    training batch. Quantify survival/progress before claiming learning is
    healthy.
  - Follow-up behavior proof:
    `curvy-looplive-proof-controllong-20260515d` is no longer running. It
    proved that checkpoints were being written and refresh checks were firing,
    but it did not prove a second full promotion-and-refresh cycle before the
    trainer stopped.
    It wrote checkpoints through `iteration_3021.pth.tar`.
  - Modal CLI must use `--env shankha-dev`; defaulting to `main` makes app log
    checks falsely report missing apps.
  - Four stale detached tournament apps from older stress/all-pairs attempts
    were asked to stop:
    `ap-hgRfLZZa1y9JS6U06hIH9k`, `ap-cIdYnGnowhAaDtn1VRQJw5`,
    `ap-uJTlKnuoh3q4uoa6UeOReu`, `ap-fiuernb2RQSnfIz9yQC0gW`.
  - Preserve for now: deployed tournament app, deployed trainer app, GIF
    browser, and fresh detached proof app `ap-eXVq2pDG90HQgKiHMcEew6`.
- Superseded Coach handoff, 2026-05-14 02:14 EDT:
  - This block is historical. Current broad launch defaults live in
    `../r18fresh_postmortem_2026-05-16/CURRENT_LAUNCH_DEFAULTS.md` and
    `src/curvyzero/contracts/curvytron.py`: `gpu-l4-t4-cpu40`, C256/N256,
    `batch_size=64`, sim8, and `browser_lines + simple_symbols + cpu_oracle`.
  - Trusted path:
    `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train --env-variant source_state_fixed_opponent`.
  - Recommended next run shape:
    `reward_variant=survival_plus_bonus_no_outcome`, `num_simulations=8`,
    `batch_size=32`, `collector_env_num=32` main, `64` bounded probe.
  - Render correction, 2026-05-15: the production policy observation surface is
    CPU `cpu_oracle` `browser_lines + simple_symbols`.
  - GPU `browser_lines + simple_symbols` remains lab/profiling-only until the
    trainer-visible contract parity gates pass.
  - `body_circles_fast + simple_symbols` is historical CPU ablation/control
    evidence only. Do not copy it as the new-run destination.
  - Historical warning from that handoff: it avoided `batch64`, multi-GPU, and
    broad `sim16`. This is superseded for the current broad L4 lane, where
    `batch_size=64` is now the default.
  - Checkpoint cadence should be around `save_ckpt_after_iter=5000-10000`, lower
    for canaries.
  - Best practical aggressive compute probe: `C256/H100/sim8` with the target
    observation surface.
  - Best cheaper wide compute probe: `C384/L4/sim8` with the target observation
    surface.
  - Keep `C32/C64` as learning-safe core because survival/tournament evidence
    supports those better than very wide collector counts.
- The main goal is still the full loop:

  ```text
  trainer writes checkpoint
  -> live run-id intake/subscriber sees checkpoint
  -> tournament rates checkpoint
  -> public leaderboard snapshot + Modal Dict pointer update
  -> Coach materializes immutable assignment + control pointer
  -> same running trainer refreshes at a safe boundary
  -> env telemetry shows the new assignment sha with provider-ok rows
  ```

- The loop is validated at all-v2 canary scale and again on the current-code
  sparse live proof. Survival improvement is a separate learning-quality claim,
  not the final link in the wiring proof. Production-scale proof remains open.
- Current tournament rankings are still not final proof. The old `51`-row
  symptom is no longer the live API shape. The active loop arenas now show 18
  standings rows from `curvy-n18conn-*` runs, while `clean3` has a huge
  375-checkpoint all-pairs round planned and `main` has the clean 18-way round.
- Do not use the stale `curvy-v2refresh18p-*` manifest for new checkpoint
  injection. Live discovery against those run ids returns zero checkpoint refs.
  Live discovery against the `curvy-n18conn-*` run ids from the `main`
  standings returns `417` checkpoint refs across 18 runs, max iteration
  `306755`.
- Injection status, 2026-05-15:
  - old `elo-loop18-live-main-20260514f` manifest now has all 417 exact refs
    seen/queued, proving the old `51` cap is not the live intake blocker;
  - that old rating manifest still has bad defaults for this purpose:
    `pair_selection=all_pairs` and `max_steps=8000`;
  - clean replacement rating manifest is
    `elo-loop18-live-main-adaptive417-20260515b`, with 417 refs,
    `adaptive_v0`, `pairs_per_round=300`, `active_pool_limit=100`,
    `games_per_pair=21`, `max_steps=1048576`,
    `decision_source_frames=1`, `decision_ms=16.6667`, GIFs on, 5 samples.
- Current training rows use H100 compute, but that does not mean GPU observation
  rendering. The active manifest used CPU-side
  `body_circles_fast + simple_symbols` observations on H100 compute; keep that
  fact as historical CPU-control evidence only.
- The production target is CPU `cpu_oracle` `browser_lines + simple_symbols`.
  GPU `browser_lines + simple_symbols` is lab/profiling-only until the
  trainer-visible contract parity gates pass.
- The running 18-row batch shows partial learning signal, but not a clean
  monotonic win.

## Survival Signal Snapshot

| Group | Rows | First mean | Best mean | Latest mean | Latest - first | Best - first | Latest up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All rows | 18 | 132.5 | 182.3 | 141.7 | +9.2 | +49.8 | 10/18 |
| Outcome only | 6 | 123.3 | 184.2 | 144.4 | +21.1 | +60.9 | 4/6 |
| Survival + bonus | 6 | 130.4 | 170.9 | 132.7 | +2.3 | +40.5 | 3/6 |
| Survival + bonus + outcome | 6 | 143.9 | 191.8 | 148.0 | +4.2 | +47.9 | 3/6 |
| Clean | 9 | 124.3 | 177.6 | 142.5 | +18.2 | +53.3 | 6/9 |
| 10% noise/skip | 9 | 140.7 | 187.0 | 140.8 | +0.2 | +46.3 | 4/9 |

Source: read-only `eval-summary` over the 18 `curvy-n18conn-*` rows on
2026-05-15 after the connected tournament/consume smoke work.

Plain read: best checkpoints improved strongly, but latest checkpoints are
noisy. The current batch does not show clean monotonic survival improvement.
Almost every row has at least one collapsed-action eval checkpoint, so stronger
policy-quality checks are still needed.

## P0 Incidents

| Incident | Current read | Required action |
| --- | --- | --- |
| v0/v2 storage split | Resolved for the active lane: exact v2 objects were deleted, recreated, verified with `version=2`, redeployed, and proven by the all-v2 canary. Non-v2 artifacts remain historical evidence only. | Keep using shared contract defaults and explicit v2 VolumeFS mapping. Audit launch manifests so stale non-v2 refs cannot sneak into a real batch. |
| Tournament ranking count | Old `51`/`90` `v2refresh18p` readings and old loop18 readings are historical/forensic, not the next launch target. The shared-contract default now points at the all-v2 canary proof until a fresh real tournament id is chosen. | Choose fresh all-v2 tournament/rating ids for the next real launch, then update the shared contract and dashboards in the same patch. |
| Tournament observation mismatch | Historical CPU diagnostic lane used `body_circles_fast + simple_symbols`, but the restart target is CPU `cpu_oracle` `browser_lines + simple_symbols`; GPU `browser_lines + simple_symbols` stays lab-only until contract parity passes. | Align trainer and tournament on the production observation surface before trusting new rankings. |
| Hidden fallback soup | Too many names can silently repair or change render settings. | Urgent patch may keep compatibility fallbacks; follow-up refactor must create one explicit observation-surface contract. |
| Running trainer refresh proof | Proven for wiring: the all-v2 canary proved clean v2 rating -> publish -> assignment -> same-trainer refresh -> provider-ok env rows. | Track scale/survival separately in `FULL_LOOP_PROOF.md`; do not reopen the wiring question unless the contract changes. |
| Old artifacts/apps clutter | Old arenas and apps make the dashboards hard to use. | Cleanup lane must preserve current lane and old champion-anchor source only. |

## Active Side Experiments

- Immortal opponent pressure: no weak-run-only live intervention for the
  restart lane. Blank/hard-coded sentinel slots are always immortal;
  checkpoint slots are mostly mortal with small explicit immortal slices; total
  immortal exposure should generally stay near `20-30%`.
- Old champion anchors: find the strongest prior full-sweep tournament winners,
  inject roughly top five exact checkpoint refs into the current clean
  tournament, and monitor acceptance/rating.
- Current batch marker: Tournament Arena must clearly default to and label the
  current arena.
- GIF speed/cap: GIFs at `80 fps` can look short even when not truncated; do
  not combine million-step max games with every-frame GIF capture without an
  explicit safe sampling policy.

## Dashboard State

- Tournament website defaults in current code point at the all-v2 canary proof
  ids. Redeploy the tournament app after the next real tournament/rating ids
  are chosen so the dashboard does not default to old loop18/v2real18 lanes.
- GIF browser “Current batch” in current code is no longer `curvy-v2real18-*`;
  it points at the restart prefix `curvy-r18v2-*`. Redeploy the GIF browser
  when the fresh batch is ready.
- Older deployed-dashboard verifications that mention `curvy-n18conn-*` or
  `curvy-loop18-live-main-20260514f` are historical and should not steer the
  next launch.

## Next Move

1. Audit/build the fresh restart18 manifest from an explicit all-v2
   leaderboard snapshot. Do not use default `/private/tmp` snapshots or old
   loop18/v2real18 ids.
2. Once fresh real tournament/rating ids are chosen, update
   `src/curvyzero/contracts/curvytron.py`, redeploy tournament/GIF apps, and
   verify the dashboards show the fresh ids/prefixes.
3. For weak-run immortal intervention, use the five rows and current mixes in
   `TRAINING_CONTROL.md`; do not mutate the live pointer until the exact audited
   assignment-writer path is selected.
4. The live feedback loop is now proven in the active all-v2 lane by
   `curvy-e2e-allv2-canary-20260515a`. `controlrun2` remains historical
   pre-reset proof.
5. Immediate plan: harden launch readiness, audit stale non-v2 refs, keep
   cleanup organized, and launch the next deliberately named larger run only
   from all-v2 inputs.
6. Keep `TODO.md` and `FULL_LOOP_PROOF.md` live as facts change.
