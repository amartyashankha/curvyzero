# TOURNAMENT_DEBUG

This doc owns tournament correctness. Do not scatter tournament bugs across
chat, `current_state.md`, and ad hoc notes.

## 2026-05-16 Clean All-205 Validation Lane

- Tournament: `curvy-r18fresh-validate-all205-20260516a`
- Rating run: `elo-r18fresh-validate-all205-20260516a`
- Detached app: `ap-VQZzMzRPLR5ZojFpN1iHbR`
- Rating call: `fc-01KRQXFK7WSS1ZEEMAW5GYAHFK`
- Seed result: exactly `205/205` checkpoint refs accepted.
- Persisted input: `round-000000/input.json`, `20,910` pairs,
  `439,110` games, all-pairs, `21` games per pair, `max_steps=1048576`.
- GIFs were disabled for this proof to avoid huge I/O; this lane proves
  submission, game execution, and ranking, not visual output.
- Liveness proof: app has hundreds of active tasks and logs show successful
  `curvytron_tournament_game` results with `ok=true`, no per-game error,
  `seat_order.mode=balanced_random`, both `swapped=true` and `swapped=false`,
  and normal terminal scores.
- Important observability caveat: with `games_per_shard=1`, the parent writes
  progress at map start and then after all game tasks return. Therefore
  `completed_game_count=0` in progress is not proof of no games. Final proof is
  `ratings.json`/`latest.json` with `205` ranked rows and
  `completed_game_count=439110`.
- 04:30 EDT poll: detached app still had `505` tasks; log tail reached pair
  indices around `1975..2034`; parsed tail had `175/175` ok games, `0` errors,
  `max_steps=1048576`, and both swapped/non-swapped seating. No final
  `ratings.json` or `latest.json` yet.
- Website/current fix: `src/curvyzero/contracts/curvytron.py` now marks this
  validation lane as the current tournament/rating id, and the deployed
  tournament website selects it by default. The UI panel formerly called
  `Rankings` now says `Leaderboard` and explicitly says rows are pending until
  the rating snapshot is written.
- Checkpoint visibility distinction: old live `latest.json` is stale at `98`
  rows and max iteration `70000`; running validation input has `205` refs and
  max iteration `140000`; running dirty/live `round-000012` input has `180` refs
  and max iteration `130000`. This means high-iteration checkpoints are in
  running tournament rounds even though completed leaderboard rows are not yet
  visible.
- Recovery hardening deployed after the latest review: a skipped stale round is
  now a consumed index for `continue_from_latest`, direct retry of that skipped
  round returns `status=skipped`, and shard-tally recovery can rebuild missing
  seat-aware win counts from stored shard games. This prevents a repaired stale
  round from being re-entered by accident. Targeted tests passed and
  `curvyzero-checkpoint-tournament-v2` was redeployed.

## 2026-05-15 Current Storage/App Contract

- Active tournament app: `curvyzero-checkpoint-tournament-v2`.
- Active tournament artifact Volume:
  `curvyzero-curvytron-tournaments-v2`, opened as Modal VolumeFS v2.
- Active checkpoint source Volume: `curvyzero-runs-v2`, opened as Modal
  VolumeFS v2.
- Active intake Dict/Queue:
  `curvyzero-curvytron-checkpoint-intake-v2` and
  `curvyzero-curvytron-checkpoint-events-v2`.
- Active public leaderboard Dict:
  `curvyzero-curvytron-opponent-leaderboard-live-v2`.
- Old tournament ids and old non-v2/hybrid storage are diagnostic history, not
  current launch truth.

## 2026-05-16 Active Live Lane

| Field | Value |
| --- | --- |
| Tournament | `curvy-r18fresh-live-20260516a` |
| Rating run | `elo-r18fresh-live-20260516a` |
| App | `curvyzero-checkpoint-tournament-v2` |
| Intended role | Live all-v2 rating service for the current 18-run scratch batch |

Plain current read:

- Completed/in-progress rounds use all-pairs over the active pool for that
  round. Uneven per-checkpoint game counts on the website are expected when
  checkpoints arrive over time. The old `71`-row dashboard symptom was stale
  `latest.json`, not a hard cap.
- As of 2026-05-16 04:03 EDT, the live lane is dirty. `latest.json` is
  `round-000008` with `98` rated checkpoints, `4,753` pairs, `99,813` games,
  `58` failed games in progress, `stable=false`, and
  `max_abs_delta=318.85214603560865`. This is useful stress evidence only.
- Root `progress.json` points at old `round-000010` as `running`
  (`12,720` pairs, `267,120` planned games, `1,068` completed games). Later
  round folders also exist: `round-000011` and `round-000012` were started by
  older code. Do not trust root progress from this lane as a clean live-service
  pointer.
- Known dirty artifacts: `round-000009` was falsely skipped by an older
  `zero_progress_smaller_pool` predicate; `round-000010`, `round-000011`, and
  `round-000012` overlap instead of forming one orderly continuation chain.
- Current deployed fixes: no-output skip now requires the real stale age floor;
  public pointers are monotonic; `continue_from_latest` intake uses one active
  claim per lane; and intake does not spawn reducer recovery for an unfinished
  running round. Focused intake/recovery/pointer tests and ruff passed before
  redeploy.
- Current intake read: manifest has `196` seen checkpoint refs and
  `updated_at=2026-05-16T08:02:52.922179Z`; Queue length reports `0`, while
  manifest `queued_checkpoint_count=196` should be treated as suspect
  bookkeeping until a clean run proves queue drain behavior.
- Next checks: stop/recheck stale detached tournament apps, then use a clean
  tournament/rating id or a deliberate artifact purge before claiming the
  tournament service is fixed at large scale. The proof target is clean latest
  -> controller-produced assignments -> same running trainer refresh/provider
  use.

## Current Tournament

Current production-source candidate is the nonzero rerate. The older 100-ref
rerate is now diagnostic only because `iteration_0` candidates reached active
top ranks.

| Field | Value |
| --- | --- |
| Tournament | `curvy-restart18-source-rerate-nonzero-20260515a` |
| Rating run | `elo-restart18-source-rerate-nonzero-20260515a` |
| App | `curvyzero-checkpoint-tournament-v2` |
| Intended role | Fresh all-v2 rerate of 96 nonzero rematerialized historical candidate refs before restart18 training |
| Source refs | `artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top96-nonzero-20260515a/refs.txt`; copied into `curvyzero-runs-v2` and audited `96/96` present |

Current status:

- Nonzero `round-000000`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=34.07017967162989`, `96` rows, `0` active
  rows. Not publishable/materializable as a trusted ranked source.
- Nonzero `round-000001`: complete, `300/300` pairs, `6300/6300` games,
  `0` failures, `stable=false`, `max_abs_delta=21.880940181012807`, `96` rows,
  `0` active rows. Not publishable/materializable as a trusted ranked source.
- Nonzero `round-000002`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=22.572625403714373`, `15` active rows and
  `81` provisional rows. Not publishable/materializable as a trusted ranked
  source.
- Nonzero `round-000003`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=39.7420779825474`, all `96` rows active,
  `0` provisional rows, `0` failures. Coverage is mature, but the stability
  gate is still failing.
- Nonzero `round-000004`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=17.371056613899057`, all `96` rows active,
  `0` provisional rows, `0` failures. Better, but still not publishable as a
  trusted ranked source.
- Nonzero `round-000005`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=15.636412948237727`, all `96` rows active,
  `0` provisional rows, `0` failures. Better, but still not publishable as a
  trusted ranked source.
- Nonzero `round-000006`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=25.199213332028748`, all `96` rows active,
  `0` provisional rows, `0` failures. Worse than round 5. Biggest mover:
  `ckpt-079-train-lightzero_exp-ckpt-iteration_240000-a391d866`, mostly from
  `random_bridge` exposure; diagnose scheduler/exposure before another round.
- Leaderboard-derived opponent-source publish/materialization remains blocked
  until a latest non-diagnostic source snapshot is `stable=true`,
  coverage-mature, published with expected round/context/roster/snapshot hashes,
  and materialized through the guarded assignment path. Bootstrap/static restart
  is not blocked by this; it can proceed from audited exact refs plus immortal
  blank/hard-coded sentinels while the tournament learns a better ordering.

Diagnostic 100-ref rerate status:

- `round-000000`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=32.58886751199381`.
- `round-000001`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=25.065565057086832`.
- `round-000002`: launched as a detached continuation,
  `fc-01KRPKQYQJGGDKBYPME1KP20BZ`; direct v2 Volume check shows full
  `100`-checkpoint roster, `previous_round_id=round-000001`, and
  `300` pairs / `6300` planned games. It later completed and advanced
  `latest.json` with `stable=false`, `max_abs_delta=23.31069784361553`.
- `round-000003`: running as detached continuation
  `fc-01KRPM5AS1TSSJMGQ961JYACBT`; direct v2 Volume input check shows full
  `100`-checkpoint roster, `previous_round_id=round-000002`, and
  `300` pairs / `6300` planned games. It later completed and advanced
  `latest.json` with `stable=false`, `max_abs_delta=24.82819365645907`.
- `round-000004`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=22.54218539334727`. Latest rank 1 is a
  nonzero checkpoint, but rank 2 is still `iteration_0`.
- `round-000005`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=19.048764303143294`. Coverage is mature
  (`games_min=567`, `distinct_opponents_min=25`), but four `iteration_0` rows
  remain active at ranks `2`, `3`, `7`, and `100`.
- `round-000006`: complete, `300/300` pairs, `6300/6300` games,
  `stable=false`, `max_abs_delta=18.39723682286698`. Coverage is mature
  (`games_min=714`, `distinct_opponents_min=29`), but `iteration_0` rows are
  ranks `1`, `2`, `7`, and `100`.
- Decision: treat this 100-ref lane as diagnostic, not restart source. It
  answered the important question: the old top-100 source pool is contaminated
  by strong `iteration_0` rows.
- Do not publish the 100-ref lane as a restart training source. It is useful
  diagnostic evidence only.

Historical note: the table below describes the invalidated v2real18 tournament
lane. Keep it as diagnostic history, not current restart guidance.

| Field | Value |
| --- | --- |
| Tournament | `curvy-v2real18-live-20260515a` |
| Rating run | `elo-v2real18-live-20260515a` plus corrected rerate TBD |
| App | `curvyzero-checkpoint-tournament-v2` |
| Intended role | Historical/diagnostic v2 real18 loop tournament for the 18-run/replacement batch |

## Current Critical Bug: Wrong Tick Duration Rerate

2026-05-15 09:26 EDT:

- The live 67-ref all-pairs rerate
  `elo-v2real18-rerate67-allpairs-20260515a` is not final evidence.
- It has successful Modal game logs, so it is useful as a liveness smoke.
- It is wrong for ranking because sampled game summaries show tournament
  runtime `source_physics_step_ms=20.0` / `decision_ms=20.0`, while the trainer
  and loaded checkpoint runtime use
  `source_physics_step_ms=16.666666666666668` /
  `decision_ms=16.666666666666668`.
- Fix requirements:
  - fail fast if tournament spec timing disagrees with consistent checkpoint
    runtime metadata;
  - launch corrected rerate with explicit
    `decision_source_frames=1`, `decision_ms=16.666666666666668`, and
    `source_physics_step_ms=16.666666666666668`;
  - for this historical diagnostic rerate, carry
    `policy_bonus_render_mode=simple_symbols` explicitly with
    `policy_trail_render_mode=body_circles_fast`;
  - include both render fields in rating context/hash/summary so future
    rankings cannot silently mix observation surfaces.
- Local implementation status: patched and tested. Focused risk tests passed,
  then the broader tournament/GIF-browser suite passed:
  `154 passed, 21 skipped`.
- Deployment status: deployed to `curvyzero-checkpoint-tournament-v2`.
- Corrected rerate status: launched detached under app
  `ap-MKU8vQNXqZWCqX6Dle0ztG`, function call
  `fc-01KRNXDVC9552230KK0KCBYZQ1`, rating id
  `elo-v2real18-rerate67-allpairs-16ms-20260515a`. Persisted input confirmed
  16.6667ms source timing, one source frame per action,
  `body_circles_fast + simple_symbols`, `max_steps=1048576`, `67`
  checkpoints, `2,211` pairs, and `46,431` games.
- Wrong-tick smoke app `ap-uIXpEjsU0Iy0lM0NHs8qEk` was stopped.

Current correction, 2026-05-15:

- 2026-05-15 03:02 EDT: clean rating `elo-loop18-live-main-adaptive417-20260515b`
  has `round-000000/input.json` and `progress.json`. The input shape is correct:
  `300` pairs / `6,300` games. Progress is still `0` completed games with phase
  `game_map_started`, and the round directory has no battle/shard outputs yet.
  This is now a game-work execution/progress issue, not a manifest-size issue.
- The web/API progress path is not currently reliable for this large live lane:
  `/api/rating-progress?fresh=1` and `curvytron_rating_progress.remote(...)`
  timed out. The code path reloads the tournament volume and may scan battle
  directories. Until fixed, prefer direct volume artifact checks:
  `ratings/<rating_run_id>/rounds/round-000000/input.json`,
  `progress.json`, `latest.json`, and battle/shard output counts.
- Modal logs and app listing must be queried with `--env shankha-dev`.
  Querying the default `main` environment gives false "app not found" results.
- The old `v2refresh18p` local manifest and docs are stale for checkpoint
  injection. Direct discovery against its listed run ids finds `0` checkpoints.
- The current live loop standings point at `curvy-n18conn-*` run ids.
- Direct discovery against those 18 run ids finds `417` exact checkpoints,
  `0` missing, max iteration `306755`.
- `curvy-loop18-live-main-20260514f` is the clean 18-way target. Its planned
  current round is 153 pairs / 3,213 games.
- `curvy-loop18-live-clean3-20260514e` is a stress/oversized lane. Its planned
  current round is 70,125 pairs / 1,472,625 games.
- All 417 exact `curvy-n18conn-*` refs were submitted to the old main intake in
  chunks of 10; after retrying the final missing 10, the manifest had
  `seen_checkpoint_count=417`.
- The old main rating id `elo-loop18-live-main-20260514f` should not be drained
  for final proof because it has `pair_selection=all_pairs` and
  `max_steps=8000`.
- A clean explicit replacement manifest now exists:
  `elo-loop18-live-main-adaptive417-20260515b`.
  It has `417` refs, `adaptive_v0`, `pairs_per_round=300`,
  `active_pool_limit=100`, `games_per_pair=21`, `max_steps=1048576`,
  `decision_source_frames=1`, `decision_ms=16.6667`, and GIFs on with 5
  evenly spaced samples.

## Problem 1: Only 51 Ranked Rows

Earlier read:

- The website shows the latest completed reduced snapshot.
- That latest completed snapshot is around `round-000004` and has `51` rows.
- A newer round exists with roughly `193` checkpoints and about `18,528`
  unordered all-pairs pair specs, which means hundreds of thousands of games.
- That newer round has not reduced into `latest.json`, so the UI still shows
  the old 51-row standings.

Prior `v2refresh18p` live API read:

- The old `v2refresh18p` arena reported completed reduced `round-000004`.
- It has `90` total standing rows.
- It completed `4,005` pairs and `84,105` games.
- Failures are `0`.
- `stable=false`, with large rating delta, so ratings are not converged truth.
- This round should still not be treated as final training truth because it
  predates the local observation-surface deploy.

Why this is bad:

- It makes the arena look stale.
- It blocks confidence that all current checkpoints are being rated.
- It suggests the continuation/scheduler may have fallen back to all-pairs
  instead of the intended bounded active-pool/adaptive behavior.

What to verify:

1. What rating spec was used for the huge round:
   - `pair_selection`;
   - `pairs_per_round`;
   - `active_pool_limit`;
   - `continue_from_latest`;
   - `round_count`;
   - roster count.
2. Whether `adaptive_v0` was intended but lost in defaults.
3. Whether `active_pool_limit` only applies to mature rows while provisional
   rows are all included.
4. Whether a live-watch drain or manual launch wrote a bad rating config.
5. Whether overlapping continuation writers reused or overwrote round ids.

Clean fix direction:

- Do not use the old 51-row diagnosis as current state without rechecking.
- Do not use the old completed 90-row `v2refresh18p` round as final proof.
- Do not use stale manifests as run-id truth; prefer live standings or fresh
  submission records, then verify by direct checkpoint discovery.
- For the 417-ref pool, use `elo-loop18-live-main-adaptive417-20260515b` as the
  rating lane. Expected first round is 300 pairs / 6,300 games.
- After render parity patch, start a clean rating run or clean continuation.
- Use explicit bounded scheduling unless the operator deliberately chooses
  all-pairs.
- Verify before launch that the generated pair count is sane.
- Verify after launch that `latest.json` advances and row count matches the
  intended pool.

## Problem 2: Policy Observation Surface Mismatch

Historical diagnostic read:

- Trainer fast rows used `body_circles_fast + simple_symbols`.
- Tournament historically carried `policy_trail_render_mode` but not
  `policy_bonus_render_mode`.
- `body_circles_fast` without the bonus mode can evaluate a policy on a surface
  different from what it trained on.
- Those tournament ratings are suspect until fixed and rerated.

Fresh production read:

- Trainer and tournament policy observations should use CPU `cpu_oracle`
  `browser_lines + simple_symbols`.
- GPU `browser_lines + simple_symbols` is lab/profiling-only until
  trainer-visible contract parity passes.
- `body_circles_fast` is historical/control only and should not be accepted as
  a fresh source-state training env policy surface.

Patch requirements:

- `policy_bonus_render_mode` is part of checkpoint identity and rating context.
- Pair specs, game specs, players, policy loader telemetry, game summary, roster
  hash, and compatibility checks must preserve it.
- Tournament policy stacks must group observations by the full
  `(trail_render_mode, bonus_render_mode)` pair.
- Historical diagnostic fast rows should evaluate on
  `body_circles_fast + simple_symbols`.
- Old artifacts may be repaired at boundaries, but new artifacts should be
  explicit.

Tests required:

- Checkpoint spec reads trail and bonus render modes from observation contract
  and training metadata.
- Pair/game specs preserve per-player bonus mode.
- `run_checkpoint_game` builds one policy stack per distinct surface pair.
- Summary records `policy_bonus_render_modes`.
- Golden or direct parity test: trainer stack and tournament stack match for
  `body_circles_fast + simple_symbols` with an active bonus.

Current test status:

- Focused tournament tests pass for checkpoint/pair/game propagation,
  per-surface stack construction, and summary `policy_bonus_render_modes`.
- The previously missing active-bonus trainer-vs-tournament parity test now
  exists locally.
- 2026-05-15 execution pass note: the local working tree already carried the
  urgent bonus propagation patch; this pass added the missing P0 regression
  coverage before claiming the patch test-complete.
- 2026-05-15 follow-up: local focused tournament tests now pass after adding:
  - active-bonus trainer/tournament stack parity for
    `body_circles_fast + simple_symbols`;
  - 51-row continuation guard showing `all_pairs` over `193` checkpoints gives
    `18,528` pairs / `389,088` games, while `adaptive_v0` with
    `pairs_per_round=300` gives `300` pairs / `6,300` games.
  Earlier command: `uv run pytest tests/test_curvytron_checkpoint_tournament.py -q`
  -> `136 passed, 11 skipped`.
- 2026-05-15 reorientation run after live-state check:
  `uv run pytest tests/test_curvytron_checkpoint_tournament.py -q`
  -> `123 passed, 11 skipped`.

## Problem 3: Max Steps And GIFs

Current intent:

- Tournament and trainer game caps should be very high, currently
  `1_048_576`, so survival is not artificially capped.
- This is explicit high cap, not a hidden infinite fallback.

Risk:

- Saving 704x704 RGB GIF frames every step for a million-step game is unsafe.

Rule:

- High max steps are fine for gameplay/eval.
- GIF capture needs an explicit safe policy: sample games, frame stride, or
  frame cap. Do not silently truncate without recording it.

## Problem 4: Eval Semantics

Current default should be:

- Tournament ratings use eval/greedy policy mode.
- Training collection may use stochastic/noisy mode.
- Rating should measure checkpoint strength under fixed rules, not exploration
  noise.

Still verify:

- one source frame per policy action;
- same action set, including straight/no-op action;
- same death/bonus rules;
- same observation surface;
- seat fairness / seat swapping across games.

## Clean Rerate Checklist

Before launch:

- Patch/tests pass.
- Rating spec printed or dry-run inspected.
- Pair count is sane.
- Observation surface says CPU `cpu_oracle` `browser_lines + simple_symbols`
  for fresh production checkpoints. Historical diagnostic rerates may name
  `body_circles_fast + simple_symbols`, but only as CPU-control evidence.
- Current arena naming/marker is clear.

After launch:

- Round input roster count matches intended candidates.
- Game summaries exist.
- Failures are zero or explicitly explained.
- `ratings.json` and `latest.json` advance.
- Website/API shows the expected current arena and row count.

## 2026-05-15 Fresh V2 Rerate

Current live rerate:

- Tournament: `curvy-v2real18-live-20260515a`
- Rating run: `elo-v2real18-rerate67-allpairs-20260515a`
- App id: `ap-uIXpEjsU0Iy0lM0NHs8qEk`
- Roster: exact `67` checkpoint refs discovered from the tracked v2 real18
  runs and replacements.
- Schedule: `all_pairs`, `2,211` pairs, `46,431` games, `21` games per pair.
- Important settings: `policy_trail_render_mode=body_circles_fast`,
  `policy_mode=eval`, `num_simulations=8`, `max_steps=1048576`, GIFs on with
  `5` evenly spaced samples per pair.

Observed state:

- The round `progress.json` on the Tournament Volume is currently stale at
  `0` completed games, but Modal logs are live and show successful game worker
  JSON records. This means the correct immediate monitor is both app logs and
  final `latest.json`, not progress alone.
- Sampled logs show `ok=true`, `error_type=null`, physical step counts such as
  `47`, `78`, `164`, and the score payload uses `max_steps=1048576`.
- Do not launch another duplicate rerate unless this app stops or logs show a
  real error. Duplicates would only add noise.
