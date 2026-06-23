# Experiment Log

## 2026-05-17

- 03:10 EDT corrected the live-loop status again:
  - `round-000015` was not a success. At `07:04Z`, `loop-status` saw active
    output: intake `956`, latest rating `414`, `919` checkpoint inputs, and
    `21` completed game summaries in the bounded probe.
  - A follow-up status at about `07:07Z` showed `round-000015` was marked
    `skipped` at `07:06:59Z` and latest rating stayed at `414`.
  - This is the current blocker: either an old in-flight drain skipped with
    stale code, or the full recovery scan still missed real output. The loop is
    not validated until the next bounded batch writes games, reduces ratings,
    publishes the trainer-facing export, and trainers load it.
  - Added code to expose compact skipped game-batch `skip_decision` details in
    `loop-status.recent_game_batches[]`. This makes the status tool answer
    "why did it skip?" without direct Modal Volume digging.
  - Focused local validation passed: `39 passed`, with expected local Modal
    execution warnings.
  - Detached `intake-drain` then exposed a separate CLI contract bug: the
    background path used `.remote()` while Modal says detached `.remote()` work
    can be canceled when the local caller exits. Patched `intake-drain` with
    `spawn_rating=true` and `wait=false` to use `.spawn()` and print the
    function-call id. Focused validation stayed green: `39 passed`.
  - Fast live `loop-status` at `07:21Z` showed latest rating advanced to
    `round-000011` / `681` checkpoints, active `round-000017` / `960`
    checkpoint inputs, intake `1063`, queue `83`, trainer refresh still
    generation `3`.
  - `loop-status` with a 4-pair activity probe at `07:22Z` found real output
    for `round-000017`: `21` completed game summaries, `1` sampled pair with
    output, no scan error. This batch is alive; next proof is rating reduction
    and trainer refresh.
  - Later status showed `round-000017` skipped with
    `scan_output_progress=false`, again from stale in-flight code. The same
    read exposed `latest.json` regressing to `round-000010` / `588`
    checkpoints even though `round-000011` / `681` had been seen.
  - Root cause for latest regression: child reduction guarded latest writes,
    but parent `rating_loop` wrote `latest.json` unconditionally after child
    return. Patched parent loop to use the same centralized latest-publish
    decision and added checkpoint-count monotonicity. Focused validation:
    `44 passed`.

- 02:41 EDT corrected stale-recovery bug:
  - App logs showed `r000012` game workers writing outputs even though compact
    progress still looked zero-started/stale. That means a stale progress file
    alone is not proof that a game batch is dead.
  - Patched the actual intake-drain recovery path to call
    `_rating_round_skip_decision(..., scan_output_progress=True)`.
  - Patched the skip decision so if the real output scan fails, it does not
    silently skip from stale control files.
  - Added focused tests for the live drain call path and scan-error behavior.
    Focused local suite passed: `36 passed`.
  - Deployed `curvyzero-checkpoint-tournament-v2` with the patched recovery
    path.
  - `round-000013` spawned with `815` checkpoint inputs after the
    continuation-pool fix. That proves the pool fix reached live state; the
    next required proof is completion/recovery, ratings, and trainer refresh.
  - Follow-up `loop-status` showed `round-000014` active with `681` checkpoint
    inputs while intake had `828`; app logs confirmed `r000014` game outputs
    were landing. Added an active-batch output probe to `loop-status` so this
    does not require log spelunking. Focused local suite passed: `37 passed`.
    Deployed the updated status tool.
  - First deployed output-scan status call was too slow because it still walked
    too much. Split the paths: recovery keeps the full safety scan before
    skipping; `loop-status` now uses a bounded activity probe over expected
    active-batch battle directories. Focused local suite remained green:
    `38 passed`. Deployed the bounded activity probe.
  - The first bounded probe still sampled too much for a fast operator check.
    Tightened `loop-status` to sample fewer pairs and stop after first real
    game output. Focused local suite remained green: `38 passed`. Deployed it.
  - Remote `loop-status` at `07:04Z` succeeded and reported active output
    directly: intake `956`, latest rating `414`, active `round-000015` with
    `919` checkpoint inputs and `21` game summaries found by the bounded
    probe. Remaining proof: batch completion/reduction, leaderboard export,
    and trainer consumption.

- 02:02 EDT deployed operator-tooling and recovery update:
  - Focused local validation passed: `34 passed`.
  - Deployed `curvyzero-checkpoint-tournament-v2` with `--mode loop-status`,
    duplicate active-game-batch backoff, and separated stale timers.
  - Intake claim stale timer remains `86400s`; zero-started tournament
    game-batch stale timer is now `600s`.
  - First remote `loop-status` showed intake `549`, latest rating `414`, queue
    `18`, trainer refresh generation `3`, and `round-000008` zero-started.
  - After redeploy plus detached drain, `round-000008` completed and wrote
    ratings.
  - Second remote `loop-status` showed latest rating at `round-000008`, current
    active internal game batch `round-000009`, intake `564`, latest rating
    `414`, queue `7`, and `150` checkpoints not yet represented in latest
    ratings. `round-000009` was about two minutes old, so it should either
    start or be recovered after `600s`.
  - Follow-up `loop-status` at `06:11Z` showed the intended recovery behavior:
    `round-000009` completed, `round-000010` was marked skipped after staying
    zero-started, and `round-000011` spawned with `681` checkpoint inputs.
    Queue length was `0`.
  - Follow-up `loop-status` at `06:26Z` showed `round-000011` had been skipped
    and `round-000012` had spawned, but with only `414` checkpoint inputs while
    intake had `705`. Root cause: continuation mode used stale
    `seen_checkpoint_refs` ahead of current `checkpoint_refs`. Patched it to
    use the current manifest pool plus seen extras; focused tests passed
    `34 passed`; redeployed.

- 01:55 EDT operator-tooling update:
  - Added local compact live-loop status path `--mode loop-status` to the
    tournament app. It reports the active manifest, intake counts, queue length,
    latest rating summary, current internal game-batch artifact, recent
    game-batch summaries, trainer refresh state, flags, and a plain next
    action.
  - This is the intended replacement for normal raw Modal Volume reads. Raw
    reads remain acceptable for deep forensics only.
  - Added duplicate active-game-batch handling so an existing input with a
    different spec returns `running_existing_round` /
    `existing_input_different_spec` instead of crashing or overwriting.
  - Bounded tournament game batch `round-000006` completed with `300` pairs,
    `6,300` games, `0` failed games, and ratings written.
  - Training-candidate auto-refresh succeeded from source `round-000007`:
    `414` rows, `100` active rows, generation `3`,
    `auto-r000007-g3-b2b6f2bc`, and `24` refresh pointers rewritten.
  - Intake later reached `545` checkpoints. Latest trainer refresh source had
    `414`, leaving about `131` newer checkpoints not yet represented in the
    trainer-facing rating source.
  - Current possible stall to verify after deploy: `round-000008` has `414`
    checkpoint inputs, `300` pairs, and `6,300` games, but the last direct read
    saw `0` started and `0` completed games.

- Current truth sync:
  - Live ids are `cz26-live-20260517a` /
    `elo-cz26-live-20260517a`; full training batch is
    `cz26-full-20260517a`.
  - Live repair is deployed. The intended active manifest config is
    `pair_selection=adaptive_v0`, `pairs_per_round=300`,
    `active_pool_limit=100`.
  - Checkpoint intake has seen `400+` cz26 checkpoints.
  - `round-000003` is only an internal tournament game-batch directory name,
    not a training round.
  - Current concise operator note:
    `CZ26_CURRENT_TRUTH_2026-05-17.md`.

- 01:36 EDT live-loop update:
  - Patched the remaining live-manifest repair hole: `_load_intake_manifest`
    now repairs valid-looking stale Modal Dict manifests before returning them,
    not just stale volume manifests. Seed, submit, and subscriber tick also use
    the repaired manifest before writing the volume artifact.
  - Targeted local suite after the final hardening patch:
    `uv run --extra dev pytest tests/test_curvytron_checkpoint_intake_repair.py`
    -> `32 passed`.
  - Redeployed `curvyzero-checkpoint-tournament-v2`.
  - Real live status after deploy:
    `checkpoint_count=414`, `seen_checkpoint_count=414`,
    `queued_checkpoint_count=414`, queue length `0`,
    `pair_selection=adaptive_v0`, `pairs_per_round=300`,
    `active_pool_limit=100`.
  - Subscriber tick committed successfully and found two fresh checkpoints,
    proving the intake watch is still live.
  - Drain skipped stale zero-progress bounded game batch `round-000005` and
    spawned rating call `fc-01KRT6GW1QABPAHZSG4TXN8RNY`.
  - Current tournament game batch is `round-000006`, with `300` pairs and
    `6,300` games. Modal logs show `r000006` games completing with `ok=true`,
    balanced random seats, and `max_steps=1048576`.
  - Root `progress.json` is not a reliable live counter during the game map; it
    can sit at zero while workers log completed games. Completion still needs
    `ratings.json`/`latest.json`.
  - After the final hardening deploy, status read `checkpoint_count=431`,
    queue length `17`, still `adaptive_v0 / 300 / 100`.

- 01:20 EDT live-loop repair:
  - Purged active intake state to only
    `manifest:cz26-live-20260517a:elo-cz26-live-20260517a`.
  - Removed stale manifest records for old e2e/r18fresh live attempts from the
    `shankha-dev` checkpoint-intake Modal dict.
  - Stopped the live ephemeral `curvyzero-*` apps from stale attempts while
    preserving deployed tournament, trainer, and browser/control apps.
  - Found the real current bug: the active `cz26` manifest was still
    `pair_selection=all_pairs`, `pairs_per_round=4950`, `games_per_pair=21`.
    That created tournament game batch `round-000003` with `4,950` pairs and
    `103,950` games. It stalled at `4` games and blocked later intake.
  - Patched/deployed stale tournament-game-batch recovery so progress/status
    rewrites no longer make a dead game batch look alive forever. Targeted
    local recovery suite: `28 passed`.
  - Repaired the live manifest in both the Modal dict and tournament volume to
    bounded continuous scheduling: `pair_selection=adaptive_v0`,
    `pairs_per_round=300`, `active_pool_limit=100`, `games_per_pair=21`,
    `games_per_shard=1`, `save_gif=false`.
  - Ran `curvytron_checkpoint_intake_drain` for the current manifest. It skipped
    the stale giant tournament game batch `round-000003` as
    `stale_incomplete_smaller_pool`, drained `100` queued checkpoint events,
    and spawned new rating call `fc-01KRT5WRCAAPSACZZYNZETR7W3`.
  - Verified new bounded `round-000004` input/progress: `300` pairs,
    `6,300` games.
  - Verified active manifest after repair: queue length `38`,
    checkpoint count `357`, `pair_selection=adaptive_v0`,
    `pairs_per_round=300`.

- Launched the real current-code batch:
  `cz26-full-20260517a`, `136` rows total (`96` Grid A + `40` Grid B).
- Build command:
  `uv run --extra dev python scripts/build_curvytron_next_batch_manifest.py --profile full --matrix-name cz26-full-20260517a --checkpoint-refs-file docs/working/training/r18fresh_postmortem_2026-05-16/TOP10_RAW_REFS_auto-r000032-g22-555c999b.txt --tournament-id cz26-live-20260517a --rating-run-id elo-cz26-live-20260517a --leaderboard-id cz26-live-20260517a-elo-cz26-live-20260517a-training`.
- Submit command:
  `uv run --extra modal python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.json --allow-launch --output artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.submit_launch.json`.
- Submit result: `136` records, all `spawned`, with train and checkpoint-poller
  function-call ids. Assignment write count: `24`. Refresh-pointer write count:
  `24`.
- Current run defaults in the manifest: v2 volumes, L4/T4 CPU40 compute,
  collector envs `256`, learner batch `64`, `num_simulations=8`,
  `max_env_step=30000000`, `max_train_iter=300000`,
  `save_ckpt_after_iter=10000`, opponent refresh every `2000` learner
  iterations, random learner seat per episode, policy surface
  `browser_lines + simple_symbols`.
- Every row uses the same old overnight rank-1 checkpoint as the initial policy
  seed. The seed is immutable and loaded in `matching_shape` mode.
- Started tournament intake for the batch under:
  tournament `cz26-live-20260517a`, rating `elo-cz26-live-20260517a`,
  leaderboard `cz26-live-20260517a-elo-cz26-live-20260517a-training`.
- Initial seed tournament completed: `4` seed checkpoints, `6` pairs, `126`
  games, `0` failed games, ratings written.
- Published the seed rating to the trainer/public leaderboard dict:
  `4` active rows, snapshot `snapshot-0beda54e3e70`.
- Ran `training-candidate-auto-refresh` with the current config ref:
  `control:training/lightzero-curvytron-visual-survival/cz26-control/attempts/try-cz26-control/opponents/training_candidate_refresh_config.json`.
  It rewrote all `24` refresh pointers from the seed leaderboard and produced
  snapshot `auto-r000000-g1-9289fc7f`.
- Historical mistake: the live intake manifest was briefly forced to explicit
  all-pairs: `pair_selection=all_pairs`, `pairs_per_round=4950`,
  `games_per_pair=21`, `games_per_shard=1`, `active_pool_limit=100`. That was
  wrong for a live stream and caused the giant stalled game batch. Current live
  config is bounded `adaptive_v0 / 300 / 100`.
- Verified all `136` top-level run directories exist in `curvyzero-runs-v2`.
  Deeper all-run status sweep is still running to distinguish active train
  attempts from queued/not-yet-started/dead calls.
- Replaced the heavy all-run status sweep with a lighter Modal Volume check.
  Result: `136/136` run trees, `136/136` attempt dirs, `136/136` train dirs,
  `136/136` rows with at least one checkpoint, and `133/136` progress-latest
  files.
- Intake status confirmed the subscriber is live: current manifest reached
  `checkpoint_count=140`. Queue was then drained by the deployed function.
- Fixed scheduler hygiene by setting the real `shankha-dev`
  `curvyzero-curvytron-checkpoint-intake-v2` active-key list to only the current
  `cz26-live-20260517a` manifest.
- Deployed drain result after cleanup: recovered a stale claim, observed latest
  rating had `43` checkpoints, and spawned a `140`-checkpoint rating
  continuation `fc-01KRT3W7QNKNH4EVZ6N66WFR9S`.
- Historical tournament game batches observed before the 01:20 repair:
  `round-000002` complete with `903` pairs and `18,963` games; `round-000003`
  running with `4,950` pairs and `103,950` games.

## 2026-05-16

- Added `NAMING_CONVENTIONS.md` and promoted it into the start-here docs. New
  current-code visible names are `cz26a` for Grid A, `cz26b` for Grid B, and
  `cz26c` for canaries. Run names use batch, row, reward outcome alpha, action
  noise, leaderboard immortality, and recipe code, while exact counts and refs
  remain structured manifest fields.
- Updated current docs to retire `survbonusout`, `so10rep10`, `rank1imm`,
  `tonight18`, `restart18`, and `r18fresh` from new launch IDs. They remain
  historical artifact labels only.
- Updated the E2E canary manifest defaults to use the `cz26c` naming contract:
  `cz26c-r001-out100-n0-imm0-b50r1` with assignment recipe `b50r1`.
- Updated `NEXT_BATCH_DESIGN.md` to use reward outcome alpha values
  `{0.0, 0.33, 0.67, 1.0}` and clarified that 24 runs is per slot recipe, not
  the full experiment once slot population is crossed.
- Added `SLOT_RECIPE_DEEP_DIVE.md` with exact r18fresh slot recipes,
  same-reward/noise slot comparisons, and candidate slot recipes for a 96-,
  120-, or 144-run crossed design.
- Expanded slot analysis with the current interpretation request: "less wall"
  means the hard-coded wall-avoidant slot fell from 10% to 5%, not that game
  walls changed; existing data cleanly compares ladder vs r2/r1 at fixed
  blank/wall weights, while B20 comparisons remain confounded. Recorded the
  two-grid idea: Grid A is the broad 24-per-slot-recipe alpha/noise/immortality
  cross; Grid B pins alpha near 0.5 with bonus on and spends budget on slot
  populations, clean/p10 noise, leaderboard immortality, and possibly opponent
  refresh cadence.
- Recorded user-proposed Grid A slot candidates: `blank50-rank1_50`,
  `blank25-wall25-rank1_50`, and a corrected complex
  `blank20-wall20-r1_30-r2_20-r3_5-r4_5` split. Added critique that
  leaderboard p10 immortality is clean but expected total immortal exposure
  varies by recipe because blank/wall are always immortal.
- Recorded planning-language decision: keep slot recipe names human-readable as
  approximate percentages, but implement recipes as exact 64-slot bags. Repeat
  each bag four times across 256 collector environments and shuffle
  deterministically.
- Updated current recommendation after a second critique pass:
  - Grid A is now the 96-run broad mixed-recipe grid: four production-like slot
    recipes crossed with four reward alphas, three noise settings, and two
    leaderboard-immortality probabilities.
  - Grid B is now the 40-run slot-focused grid: pure blank, pure wall-avoidant,
    pure rank1, user-proposed coarse mixtures, and anchor mixed recipes crossed
    with clean/p10 and p0/p10.
  - Ladder can expand Grid A to 120 runs if tournament robustness becomes more
    important than keeping Grid A around 100.
- Added `GRID_REFINEMENT_REVIEW.md` as the latest high-level synthesis for the
  two-grid design.
- User locked scope: Grid A is 96 runs and Grid B is 40 runs. Removed live
  optional-expansion language from the planning docs; ladder stays in Grid B.
- Recorded opponent recipe count contract: author recipes as a 64-slot bag,
  repeat the bag four times across 256 collector environments, shuffle
  deterministically, and keep learner `batch_size=64` unchanged. Exact
  mini-batch proportions are a future stratified replay feature, not part of
  this design.
- Implementation read: `scripts/build_curvytron_tonight18_manifest.py` already
  uses 64-slot recipe counts for the old r18fresh-style matrix, but it still
  emits the historical 18-row shape. It must be replaced or extended before it
  can launch the locked Grid A 96-row and Grid B 40-row batches.
- Created fresh r18fresh postmortem workspace.
- Archived old planning context by policy only; no old files moved.
- Current priorities:
  - Deep per-run analysis.
  - Tournament analysis.
  - Stop/continue decision for the 18-run batch.
  - Keep `ownlatest` `20260516b` control running.
  - Redesign next batch after evidence review.
  - Cleanup stale apps/artifacts after analysis.
- Closed completed subagent lanes after collecting reports on training signal,
  tournament state, top-100/top-10 interpretation, data sources, and testing
  gaps.
- Direct read of trainer/public leaderboard snapshot:
  `auto-r000032-g22-555c999b`, generation `22`, `563` rows, `100` active rows.
- Direct read of rating latest:
  `round-000033`, `564` rows, `100` active, `464` retired, `stable=false`,
  `max_abs_delta=40.84`.
- Current rank 1 remains a mid-run plus-outcome checkpoint:
  `ckpt-432-train-lightzero_exp-ckpt-iteration_180000-0ed114de`.
- Planning update: preserve raw top 10 for audit, but next launch now uses only
  the old-overnight rank-1 checkpoint as the shared initial policy seed for all
  Grid A/Grid B/canary trainers.
- Fast local validation sweep passed: `45 passed in 2.47s`.
- Targeted full-loop component tests passed: `14 passed in 1.16s`, with three
  expected local Modal warnings.
- Raw top-10 checkpoint refs preserved in
  `TOP10_RAW_REFS_auto-r000032-g22-555c999b.txt`.
- Computed matched-grid analysis over common `0..240k` eval checkpoints and
  wrote `MATCHED_GRID_ANALYSIS.md`.
- Rechecked trainer/public leaderboard after rating had reached round 33:
  public snapshot is still `auto-r000032-g22-555c999b`, generation `22`, with
  `563` rows and `100` active rows.
- Pulled latest rating again; current detailed stocktake uses `round-000035`,
  `573` rows, `100` active, `stable=false`, `max_abs_delta=87.80`.
- Wrote `DETAILED_RUN_STOCKTAKE.md` with per-run survival shape, best
  tournament checkpoint, latest tournament row, top-100 counts, and knob-level
  tournament/survival readout.
- Added `REWARD_BREAKDOWN_ANALYSIS.md`. Reward fields confirm that
  `mean_training_reward` is variant-specific, bonus pickup was nearly absent,
  plus-outcome's terminal outcome must be inferred as a residual, and latest
  reward is near own best in only `1/18` runs.
- Joined `572/573` current rating rows to eval checkpoints. Rating correlates
  moderately with survival (`0.431` all rows, `0.302` active rows) but not with
  own reward overall. This means tournament rank, survival eval, and own reward
  should remain separate seed-selection views.
- Dispatched four parallel analysis lanes: existing tooling, matched knob
  comparison, tournament/Elo behavior, and local tool inventory.
- Confirmed `scripts/analyze_curvytron_eval_curves.py` handles
  `mean_survival` and `mean_training_reward` once the Modal log wrapper is
  stripped from `/tmp/r18fresh_eval_status.json`.
- Created `/tmp/r18fresh_eval_status_clean.json` and
  `/tmp/r18fresh_eval_curve_scores.json` for local analysis artifacts.
- Added `TREND_ANALYSIS.md` with matched survival, own reward, tournament, and
  checkpoint-throughput readout. The trend conclusion is retention failure
  after real intermediate learning, not total absence of signal.
- Tournament lane reported duration trend: round 0 mean physical steps `131.16`,
  round 35 mean `162.20`, first five rounds `131.59`, last five `159.78`,
  correlation with round index `0.945`.
- Found `curvy-ownlatest-staticmix-20260516b` control lane. Only selected rows
  r007/r009/r011 were launched; all are clean `survival_plus_bonus_no_outcome`
  rows with `own_checkpoint_opponent_refresh_enabled=true`.
- Pulled `/tmp/ownlatestb_selected3_eval_status.raw`, cleaned it to
  `/tmp/ownlatestb_selected3_eval_status_clean.json`, and scored it with
  `analyze_curvytron_eval_curves.py`. The control is early but already shows
  the same retention shape: best checkpoints are better than latest
  checkpoints.
- Reviewed optimizer docs and the newest optimizer update. Wrote
  `H100_L4_OPTIMIZER_HANDOFF.md`. Current conclusion: r18fresh used the correct
  production policy surface (`browser_lines + simple_symbols`) through
  `cpu_oracle` on H100 compute; it did not use GPU observation rendering.
  Float32 is now the aggressive candidate for the future batched GPU observation
  boundary, while scalar `policy_observation_backend=jax_gpu` remains out of
  production.
- Integrated the fresh optimizer recommendation: best current-surface L4 row was
  `713.83` env steps/s at C256/batch64, best H100 row was `1001.94` env steps/s
  at C256/batch32. L4 throughput is about `28.8%` lower, which is acceptable for
  cheaper broad runs. Current broad default is now L4/C256/N256/batch64/sim8
  with `browser_lines + simple_symbols + cpu_oracle`.
- Deployed E2E canary artifact check: local synthetic loop is green, and the
  remote canary produced checkpoints plus rating rounds through at least
  `round-000040`. The remote full loop is not proven: the public/trainer
  leaderboard snapshot was stale at generation `0`, and the candidate refresh
  lineage rewrote the `e2e-rankslot-proof-control-20260516a` pointer rather
  than the pointer watched by `curvy-e2e-current-contract-20260516a`.
- Cleanup pass after that finding: stopped all active ephemeral
  `curvyzero-checkpoint-tournament-v2` apps that were still burning compute
  from stale r18fresh/scale/canary attempts. Kept the deployed tournament,
  trainer, and GIF-browser apps running. Preserved v2 volumes and the r18fresh
  overnight artifacts.
- Narrow volume cleanup: deleted only transient tournament scale probes
  `curvy-scale-probe-18latest-gamefanout-20260516a`,
  `curvy-scale-probe-5latest-gamefanout-20260516a`, and
  `curvy-scale-probe-5latest-nogif-20260516a`. Did not delete r18fresh,
  bounded r18fresh, e2e proof artifacts, training runs, or control pointers.
- 2026-05-17 03:50-04:01 EDT live-control cleanup:
  - Added `scripts/curvytron_live_loop_control.py`, a local operator tool that
    calls deployed Modal functions by name. This replaces normal live use of
    `modal run` for status/control.
  - Reason: `modal run` creates a temporary app with scheduled functions; one
    such temporary app fired drain logic while stopping and produced
    `ConflictError: The app is stopped or disabled`.
  - Focused local status/control validation passed: `43 passed`.
  - Deployed `curvyzero-checkpoint-tournament-v2`.
  - Deployed-app status at `08:01Z`: active game batch `round-000020`, `1228`
    checkpoint inputs, `593` pairs, `12453` games, bounded config
    `adaptive_v0 / 300 / 100`, real output found by activity probe (`21`
    completed summaries), latest rating still `round-000010` / `588`, trainer
    refresh still generation `3`.
  - Follow-up: wait for `round-000020` to complete/reduce. Success requires
    latest rating to advance beyond `round-000010` / `588` and trainer refresh
    to advance beyond generation `3`.
- 2026-05-17 04:05-04:08 EDT:
  - Deployed-app script status still shows `round-000020` active, but output
    has not visibly advanced beyond one completed pair: `21` summaries,
    output age about `345s` at `08:05Z`, about `497s` at `08:07Z`.
  - Integrated critique-lane cleanup: docs now point live status/control to
    `scripts/curvytron_live_loop_control.py`, and local-entrypoint live modes
    now raise instead of steering operators toward `modal run`.
  - Redeployed `curvyzero-checkpoint-tournament-v2` after focused guard tests:
    `6 passed`.
- 2026-05-17 04:08 EDT:
  - App logs show many tournament game inputs canceled at the Modal function
    timeout of `1800s`. This is the hidden cap that explains active batches
    with a little output and then no progress.
  - Local patch raises tournament game, game-shard, pair, and rating-round
    function timeouts to `24h`. This does not rescue already-launched old-code
    workers, but it should prevent the next game batch from being killed by the
    old 30-minute game timeout.
- 2026-05-17 04:29 EDT:
  - Safe deployed-app status still shows the loop blocked before rating publish.
    Active game batch is `round-000022` with `1361` checkpoint inputs, `659`
    pairs, `13839` games, and `0` probe-visible completed game summaries.
  - Intake is alive and growing (`1381` checkpoints, `793` missing from latest
    rating, queue length `85`), but latest rating is still `round-000010` /
    `588` and trainer refresh is still generation `3` from `round-000007`.
  - App logs include old `1800s` cancellations and newer `600s` cancellations.
    The `600s` path is not yet understood. Next investigation target is which
    deployed function/input still has a 10-minute timeout or whether
    `round-000022` started on pre-patch code.
  - At `08:32Z`, safe `drain-if-ready` observed no active game batch and spawned
    exactly one bounded drain, function call
    `fc-01KRTH2CJ483F5RYV945MZ4BQH`. Immediate follow-up is to verify that this
    creates the next active game batch and writes game summaries.
  - Added a script-side operator drain-request lease to
    `scripts/curvytron_live_loop_control.py` and byte-compiled it. Purpose:
    avoid duplicate operator drain spawns during Modal scheduling delay.
  - `08:39Z`: prior drain request still had not created an active batch. Queue
    length had grown to `482`. Guarded `drain-if-ready` spawned
    `fc-01KRTHECNW3S4EMBPZ5A3QX6JE` and wrote the operator lease.
  - Refined the safe control script again: future drain calls wait for the
    deployed drain result by default and print a compact summary, with
    `--detach-drain` kept as the explicit fire-and-forget option. Byte-compile
    passed.
  - `08:42Z`: status showed queue length `0` and no active game batch. The
    earlier detached drain appears to have consumed queue events without giving
    the operator its result.
  - `08:43Z`: synchronous deployed drain with `--ignore-drain-request-lease`
    returned a useful result: stale rating claim repaired, `1494` checkpoint
    rating pool, `906` desired new checkpoints, rating loop spawned as
    `fc-01KRTHMV0DN5TH8SBZM6EM945Y`.
  - `08:44Z`: status still showed no active game batch. Queue had only `4`
    events, so queue draining is not the immediate bottleneck. No logs were
    available for `fc-01KRTHMV0DN5TH8SBZM6EM945Y` through `modal app logs
    --function-call ... --tail 50`.
  - Patched scheduled drain tick to share the feedback-loop readiness gate. It
    now computes status, calls `_feedback_loop_control_decision(...,
    action="drain-if-ready")`, and only spawns drain work when that decision is
    positive. Focused tests: `2 passed`. Byte-compile passed. Deployed
    `curvyzero-checkpoint-tournament-v2`.
  - Post-deploy system log tail still shows old function-id `600s`
    cancellations. Treat these as likely pre-deploy scheduled tick inputs until
    the old 10-minute window has cleared; re-check after `08:58Z`.
  - Added/updated focused tests for scheduled drain tick gating. Validation:
    `4 passed`, `2 warnings` from local Modal execution. Redeployed
    `curvyzero-checkpoint-tournament-v2` again with the tested patch.

## Entry Template

Date/time:
Run(s):
Artifacts:
Command(s):
Observation:
Decision/follow-up:
