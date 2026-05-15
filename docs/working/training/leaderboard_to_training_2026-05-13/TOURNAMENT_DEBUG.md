# TOURNAMENT_DEBUG

This doc owns tournament correctness. Do not scatter tournament bugs across
chat, `current_state.md`, and ad hoc notes.

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

## Current Tournament

Historical note: the table below describes the invalidated v2real18 tournament
lane. After the all-v2 reset, no tournament in the recreated v2 Volume is
current yet. The next current tournament should be the fresh all-v2 deployed
canary, with new tournament/rating ids and durable proof in this doc.

| Field | Value |
| --- | --- |
| Tournament | `curvy-v2real18-live-20260515a` |
| Rating run | `elo-v2real18-live-20260515a` plus corrected rerate TBD |
| App | `curvyzero-checkpoint-tournament-v2` |
| Intended role | Current v2 real18 loop tournament for the 18-run/replacement batch |

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
  - carry `policy_bonus_render_mode=simple_symbols` explicitly with
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

Current read:

- Trainer fast rows use `body_circles_fast + simple_symbols`.
- Tournament historically carried `policy_trail_render_mode` but not
  `policy_bonus_render_mode`.
- `body_circles_fast` without the bonus mode can evaluate a policy on a surface
  different from what it trained on.
- Current tournament ratings are suspect until fixed and rerated.

Patch requirements:

- `policy_bonus_render_mode` is part of checkpoint identity and rating context.
- Pair specs, game specs, players, policy loader telemetry, game summary, roster
  hash, and compatibility checks must preserve it.
- Tournament policy stacks must group observations by the full
  `(trail_render_mode, bonus_render_mode)` pair.
- Current fast rows should evaluate on `body_circles_fast + simple_symbols`.
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
